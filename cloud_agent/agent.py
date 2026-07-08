import asyncio
import inspect
import logging
import sqlite3
import uuid
from typing import Any, List, Optional, Dict, Literal, TypedDict, Annotated
from pydantic import BaseModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore
from langgraph.store.base import BaseStore
from langgraph.prebuilt import ToolNode, InjectedStore
from langgraph.types import interrupt

from cloud_agent.config import settings
from cloud_agent.connection import manager

logger = logging.getLogger("cloud_agent.agent")

scheduled_tasks: set[asyncio.Task] = set()


async def schedule_task(callback, delay_seconds: float, *, task_name: Optional[str] = None):
    """Run a coroutine or callable after a delay without blocking the caller."""
    await asyncio.sleep(delay_seconds)

    if inspect.iscoroutinefunction(callback):
        await callback()
    else:
        result = callback()
        if inspect.isawaitable(result):
            await result

    return {
        "status": "completed",
        "task_name": task_name or getattr(callback, "__name__", "scheduled_task"),
    }


# ---------------------------------------------------------------------------
# 1. Persistence
# ---------------------------------------------------------------------------

conn = sqlite3.connect("agent_memory.db", check_same_thread=False)

store = SqliteStore(conn)
store.setup()

checkpointer = SqliteSaver(conn)
checkpointer.setup()

# ---------------------------------------------------------------------------
# 2. Long-term memory
# ---------------------------------------------------------------------------

class UserProfile(BaseModel):
    name: Optional[str] = None
    preferences: dict = {}
    notes: list[str] = []

def get_profile(user_id: str) -> dict:
    existing = store.get(("profile", user_id), "data")
    return existing.value if existing else UserProfile().model_dump()

def put_profile(user_id: str, profile: dict) -> None:
    store.put(("profile", user_id), "data", profile)

@tool
def save_memory(
    user_id: str,
    field: str,
    key: Optional[str],
    value: str,
    *,
    store: Annotated[BaseStore, InjectedStore()],
) -> str:
    """Save a fact or preference about the user for future conversations.
    field='name' for the user's name, field='preference' with a key
    (e.g. 'tone', 'timezone') for stated preferences, field='note' for
    any other durable fact worth remembering."""
    profile = get_profile(user_id)
    if field == "name":
        profile["name"] = value
    elif field == "preference" and key:
        profile["preferences"][key] = value
    else:
        profile["notes"].append(value)
    put_profile(user_id, profile)
    return f"Saved: {field} = {value}"

# ---------------------------------------------------------------------------
# 3. Control-flow tools
# ---------------------------------------------------------------------------

@tool
def ask_user(question: str) -> str:
    """Ask the user ONE clarifying question when information required to
    complete their request is missing and can't be reasonably assumed.
    Execution pauses until the user replies; the reply is returned to you."""
    return interrupt({"question": question})

@tool
def final_answer(summary: str, result: dict) -> str:
    """Call this exactly once, when the request is fully handled, to end
    the turn. `summary` is a short human-readable sentence. `result` is a
    free-form JSON object holding whatever structured data fits this
    specific request — its shape can vary by scenario."""
    return "finalized"

# ---------------------------------------------------------------------------
# 4. Device domain tool
# ---------------------------------------------------------------------------

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


@tool
async def schedule_delayed_task(task_description: str, delay_minutes: int = 30) -> str:
    """Schedule a reminder or follow-up task to run after the requested delay in minutes."""
    if delay_minutes < 0:
        return "Error: delay_minutes must be zero or greater."

    async def _run_task() -> None:
        await asyncio.sleep(delay_minutes * 60)
        logger.info("Scheduled task executed after %s minute(s): %s", delay_minutes, task_description)
        return {"message": task_description, "delay_minutes": delay_minutes}

    task = asyncio.create_task(_run_task())
    scheduled_tasks.add(task)
    task.add_done_callback(scheduled_tasks.discard)
    return f"Scheduled task: {task_description} in {delay_minutes} minute(s)."


DOMAIN_TOOLS = [execute_device_command, schedule_delayed_task]
CONTROL_TOOLS = [save_memory, ask_user, final_answer]
ALL_TOOLS = DOMAIN_TOOLS + CONTROL_TOOLS

# ---------------------------------------------------------------------------
# 5. Mock LLM to allow complete offline execution without API keys
# ---------------------------------------------------------------------------

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
            # We just completed the tool execution. Return a call to final_answer
            tool_output = tool_messages[-1].content
            tool_calls = [
                {
                    "name": "final_answer",
                    "args": {
                        "summary": "Completed tool execution successfully.",
                        "result": {"output": tool_output}
                    },
                    "id": f"call_mock_{uuid.uuid4().hex[:6]}",
                    "type": "tool_call"
                }
            ]
            response_message = AIMessage(
                content="Tool execution is complete. Finalizing response.",
                tool_calls=tool_calls
            )
        else:
            # Inspect last human message to decide tool usage
            last_msg = ""
            for msg in reversed(messages):
                if msg.type == "human" or getattr(msg, "role", None) == "user":
                    last_msg = msg.content.lower()
                    break
            
            # Check for memory storage request
            if "remember" in last_msg or "save preference" in last_msg or "my name is" in last_msg:
                field = "name" if "name" in last_msg else "note"
                val = last_msg.replace("remember", "").replace("my name is", "").replace("save preference", "").strip()
                tool_calls = [
                    {
                        "name": "save_memory",
                        "args": {
                            "user_id": "default_user",
                            "field": field,
                            "key": "pref" if "preference" in last_msg else None,
                            "value": val
                        },
                        "id": f"call_mock_{uuid.uuid4().hex[:6]}",
                        "type": "tool_call"
                    }
                ]
                response_message = AIMessage(
                    content="Executing save memory tool...",
                    tool_calls=tool_calls
                )
            elif "schedule" in last_msg or "remind" in last_msg or "task" in last_msg:
                import re

                delay_minutes = 30
                match = re.search(r"(\d+)\s*(minute|minutes|min|mins|hour|hours|hr|hrs)", last_msg)
                if match:
                    value = int(match.group(1))
                    unit = match.group(2).lower()
                    delay_minutes = value * 60 if unit.startswith("hour") else value

                tool_calls = [
                    {
                        "name": "schedule_delayed_task",
                        "args": {
                            "task_description": last_msg,
                            "delay_minutes": delay_minutes,
                        },
                        "id": f"call_mock_{uuid.uuid4().hex[:6]}",
                        "type": "tool_call"
                    }
                ]
                response_message = AIMessage(
                    content="Scheduling a delayed task...",
                    tool_calls=tool_calls
                )
            else:
                # Rule-based routing to show tool execution
                # Check if the user is asking to execute/run a command or check device parameters
                device_keywords = ["run", "execute", "device", "disk", "cpu", "memory", "system", "command", "ls", "uname", "df", "free", "dir"]
                
                if any(kw in last_msg for kw in device_keywords):
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

                    tool_calls = [
                        {
                            "name": "execute_device_command",
                            "args": {"command": command_to_run},
                            "id": f"call_mock_{uuid.uuid4().hex[:6]}",
                            "type": "tool_call"
                        }
                    ]
                    response_message = AIMessage(
                        content="Checking target device. Executing tool command...",
                        tool_calls=tool_calls
                    )
                else:
                    # Greet user / answer directly using final_answer
                    tool_calls = [
                        {
                            "name": "final_answer",
                            "args": {
                                "summary": "Direct response to query.",
                                "result": {"message": f"Hello! I am the Cloud Agent. I see your request: '{last_msg}'. Since no system or device commands were requested, I have responded directly."}
                            },
                            "id": f"call_mock_{uuid.uuid4().hex[:6]}",
                            "type": "tool_call"
                        }
                    ]
                    response_message = AIMessage(
                        content="Answering query directly...",
                        tool_calls=tool_calls
                    )
            
        generation = ChatGeneration(message=response_message)
        return ChatResult(generations=[generation])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._generate(messages, stop, run_manager, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "mock-device-router-llm"

    def bind_tools(self, tools: List[Any], **kwargs: Any) -> "MockChatModel":
        self.bound_tools = tools
        return self

# ---------------------------------------------------------------------------
# 6. LLM Configuration
# ---------------------------------------------------------------------------

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

def get_llm_with_tools():
    llm = get_llm()
    provider = settings.LLM_PROVIDER.lower()
    if provider == "openai" and settings.OPENAI_API_KEY:
        return llm.bind_tools(ALL_TOOLS, tool_choice="required")
    else:
        return llm.bind_tools(ALL_TOOLS)


llm_with_tools = get_llm_with_tools()

# ---------------------------------------------------------------------------
# 7. Graph state + agent node
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: list
    output: Optional[dict]

SYSTEM_PROMPT = """You are a general-purpose assistant that can handle any
kind of request (planning, answering questions, drafting things, etc).

Rules:
- Use device tools (like execute_device_command) to gather whatever real status or information the request needs.
- If something material is missing and you can't reasonably assume it,
  call ask_user with ONE specific question. Don't ask about things you can
  infer or that don't materially change the outcome.
- If the user shares a durable fact/preference/name, call save_memory.
- If the user wants a reminder or delayed follow-up task, call schedule_delayed_task with a clear task description and delay in minutes.
- When the request is fully handled, call final_answer exactly once with
  a short summary and a result object shaped appropriately for this
  specific request.

Known user info: {profile}
"""

async def agent_node(state: AgentState, config, *, store: BaseStore):
    user_id = config["configurable"].get("user_id", "default_user")
    profile = get_profile(user_id)
    system = {"role": "system", "content": SYSTEM_PROMPT.format(profile=profile)}
    response = await llm_with_tools.ainvoke([system] + state["messages"])
    return {"messages": [response]}

def route_after_agent(state: AgentState):
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    if any(tc["name"] == "final_answer" for tc in tool_calls):
        return "finalize"
    return "tools"

def finalize(state: AgentState):
    last = state["messages"][-1]
    call = next(tc for tc in last.tool_calls if tc["name"] == "final_answer")
    
    summary = call["args"].get("summary", "Finalized.")
    result_data = call["args"].get("result", {})
    
    # Format a friendly response string for client display
    if isinstance(result_data, dict):
        output_txt = result_data.get("output") or result_data.get("message")
        if output_txt:
            content = f"{summary}\n\n{output_txt}"
        else:
            import json
            content = f"{summary}\n\n{json.dumps(result_data, indent=2)}"
    else:
        content = f"{summary}\n\n{result_data}"

    tool_message = ToolMessage(content=content, tool_call_id=call["id"])
    return {"messages": [tool_message], "output": call["args"]}

# ---------------------------------------------------------------------------
# 8. Graph assembly
# ---------------------------------------------------------------------------

builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(ALL_TOOLS))
builder.add_node("finalize", finalize)

builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent", route_after_agent, {"tools": "tools", "finalize": "finalize"}
)
builder.add_edge("tools", "agent")
builder.add_edge("finalize", END)

graph = builder.compile(checkpointer=checkpointer, store=store)

# ---------------------------------------------------------------------------
# 9. Graph wrapper for backwards capability and default session IDs
# ---------------------------------------------------------------------------

class CompiledGraphWrapper:
    def __init__(self, inner_graph):
        self.inner_graph = inner_graph

    async def ainvoke(self, input, config=None, **kwargs):
        if config is None:
            config = {}
        if "configurable" not in config:
            config["configurable"] = {}
        if "thread_id" not in config["configurable"]:
            config["configurable"]["thread_id"] = "default-thread"
        if "user_id" not in config["configurable"]:
            config["configurable"]["user_id"] = "default_user"
        return await self.inner_graph.ainvoke(input, config, **kwargs)

    def invoke(self, input, config=None, **kwargs):
        if config is None:
            config = {}
        if "configurable" not in config:
            config["configurable"] = {}
        if "thread_id" not in config["configurable"]:
            config["configurable"]["thread_id"] = "default-thread"
        if "user_id" not in config["configurable"]:
            config["configurable"]["user_id"] = "default_user"
        return self.inner_graph.invoke(input, config, **kwargs)

    def __getattr__(self, name):
        return getattr(self.inner_graph, name)

compiled_graph = CompiledGraphWrapper(graph)
