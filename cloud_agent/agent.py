import logging
import asyncio
from typing import Any, List, Optional, Dict, Literal
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

from cloud_agent.config import settings
from cloud_agent.connection import manager

logger = logging.getLogger("cloud_agent.agent")

# Define Tool
@tool
async def execute_device_command(command: str, config: Optional[dict] = None) -> str:
    """
    Executes a shell/terminal command on the remote Linux Device Agent.
    Use this for checking environment details, disk usage, executing scripts, file details,
    cpu/ram metrics, and running standard bash utilities on the connected local device.
    """
    if not manager.active_devices:
        return "Error: No device agent is currently connected to the cloud agent."
    
    # Send to the first active device for this sandbox/session
    device_id = list(manager.active_devices.keys())[0]
    
    # Retrieve the progress callback (if setup via LangGraph execution context)
    progress_cb = None
    if config:
        configurable = config.get("configurable", {})
        progress_cb = configurable.get("progress_callback", None)
        
    try:
        # Dispatch command and wait for response
        logger.info(f"Tool executing command: '{command}' on device '{device_id}'")
        output = await manager.send_command_and_wait(
            device_id=device_id, 
            command=command, 
            progress_cb=progress_cb
        )
        return output
    except Exception as e:
        return f"Error executing command: {str(e)}"

# Define Mock LLM to allow complete offline execution without API keys
class MockChatModel(BaseChatModel):
    bound_tools: List[Any] = []

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Check if a tool has already run (we have a ToolMessage in history)
        tool_messages = [msg for msg in messages if isinstance(msg, ToolMessage)]
        
        if tool_messages:
            # We just completed the tool execution. Return the final output message.
            tool_output = tool_messages[-1].content
            response_message = AIMessage(
                content=f"Device execution completed successfully. Here is the output:\n\n{tool_output}"
            )
        else:
            # Inspect last human message to decide tool usage
            last_msg = messages[-1].content.lower() if messages else ""
            
            # Rule-based routing to show tool execution
            # Check if the user is asking to execute/run a command or check device parameters
            device_keywords = ["run", "execute", "device", "disk", "cpu", "memory", "system", "command", "ls", "uname", "df", "free", "dir"]
            
            # Simple parser to extract potential shell commands
            command_to_run = "uname -a"  # default
            if "ls" in last_msg or "dir" in last_msg:
                command_to_run = "dir" if "dir" in last_msg else "ls"
            elif "disk" in last_msg or "df" in last_msg:
                command_to_run = "df -h"
            elif "memory" in last_msg or "free" in last_msg:
                command_to_run = "free -m"
            elif "cpu" in last_msg:
                command_to_run = "lscpu"
            elif "execute" in last_msg or "run" in last_msg:
                # try to extract whatever command is inside quotes
                import re
                quotes = re.findall(r'"([^"]*)"', last_msg)
                if quotes:
                    command_to_run = quotes[0]
                else:
                    quotes_single = re.findall(r"'([^']*)'", last_msg)
                    if quotes_single:
                        command_to_run = quotes_single[0]

            tool_calls = []
            # If keywords match, generate tool call
            if any(kw in last_msg for kw in device_keywords):
                tool_calls = [
                    {
                        "name": "execute_device_command",
                        "args": {"command": command_to_run},
                        "id": "call_mock_123456",
                        "type": "tool_call"
                    }
                ]
                response_message = AIMessage(
                    content="Checking target device. Executing tool command...",
                    tool_calls=tool_calls
                )
            else:
                response_message = AIMessage(
                    content=f"Hello! I am the Cloud Agent. I see your request: '{messages[-1].content}'. Since no system or device commands were requested, I have responded directly without using the device tool. How else can I assist you with your device agent?"
                )
            
        generation = ChatGeneration(message=response_message)
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        return "mock-device-router-llm"

    def bind_tools(self, tools: List[Any], **kwargs: Any) -> "MockChatModel":
        self.bound_tools = tools
        return self

def get_llm():
    """Initializes the LLM according to system configuration."""
    provider = settings.LLM_PROVIDER.lower()
    if provider == "openai":
        if not settings.OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY is not set. Falling back to Mock LLM.")
            return MockChatModel()
        return ChatOpenAI(model="gpt-4o-mini", api_key=settings.OPENAI_API_KEY)
    elif provider == "gemini":
        if not settings.GEMINI_API_KEY:
            logger.warning("GEMINI_API_KEY is not set. Falling back to Mock LLM.")
            return MockChatModel()
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=settings.GEMINI_API_KEY)
    else:
        logger.info("Using offline Mock Chat Model for device routing.")
        return MockChatModel()

# Assemble custom LangGraph React Agent
tools = [execute_device_command]
tool_node = ToolNode(tools)

# Bind tools to the LLM agent
llm = get_llm()
llm_with_tools = llm.bind_tools(tools)

# Node 1: Call LLM agent
async def call_model(state: MessagesState):
    messages = state['messages']
    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}

# Conditional routing edge
def should_continue(state: MessagesState) -> Literal["tools", "__end__"]:
    messages = state['messages']
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return "__end__"

# Build state graph
workflow = StateGraph(MessagesState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

compiled_graph = workflow.compile()
