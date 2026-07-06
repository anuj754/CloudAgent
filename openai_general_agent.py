"""
General-purpose LangGraph agent — OpenAI version.
Same architecture as the Anthropic version: memory, ask_user (interrupt/resume),
final_answer (structured JSON envelope). Only the model + tool_choice value differ.

Install:
    pip install langgraph langgraph-checkpoint-sqlite langchain-openai pydantic
Env:
    export OPENAI_API_KEY=...
"""

import sqlite3
import uuid
from typing import TypedDict, Optional, Annotated
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore
from langgraph.store.base import BaseStore
from langgraph.prebuilt import ToolNode, InjectedStore
from langgraph.types import interrupt, Command


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
# 4. Example domain tools — add more here without touching the graph
# ---------------------------------------------------------------------------

@tool
def get_calendar_events(date: str) -> list:
    """Get the user's calendar events for a given date (YYYY-MM-DD)."""
    return []


@tool
def get_tasks(date: str) -> list:
    """Get the user's open tasks/todos relevant to a given date (YYYY-MM-DD)."""
    return []


DOMAIN_TOOLS = [get_calendar_events, get_tasks]
CONTROL_TOOLS = [save_memory, ask_user, final_answer]
ALL_TOOLS = DOMAIN_TOOLS + CONTROL_TOOLS


# ---------------------------------------------------------------------------
# 5. Graph state + agent node
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: list
    output: Optional[dict]


# NOTE: OpenAI's forced tool-call value is "required" (not "any" like
# Anthropic/Gemini). This is the one real behavioral difference across providers.
llm = ChatOpenAI(model="gpt-5.5").bind_tools(ALL_TOOLS, tool_choice="required")

SYSTEM_PROMPT = """You are a general-purpose assistant that can handle any
kind of request (planning, answering questions, drafting things, etc).

Rules:
- Use domain tools to gather whatever real information the request needs.
- If something material is missing and you can't reasonably assume it,
  call ask_user with ONE specific question. Don't ask about things you can
  infer or that don't materially change the outcome.
- If the user shares a durable fact/preference/name, call save_memory.
- When the request is fully handled, call final_answer exactly once with
  a short summary and a result object shaped appropriately for this
  specific request.

Known user info: {profile}
"""


def agent_node(state: AgentState, config, *, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    profile = get_profile(user_id)
    system = {"role": "system", "content": SYSTEM_PROMPT.format(profile=profile)}
    response = llm.invoke([system] + state["messages"])
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
    tool_message = ToolMessage(content="Finalized.", tool_call_id=call["id"])
    return {"messages": [tool_message], "output": call["args"]}


# ---------------------------------------------------------------------------
# 6. Graph assembly
# ---------------------------------------------------------------------------

builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(ALL_TOOLS))
builder.add_node("finalize", finalize)

builder.add_edge("__start__", "agent")
builder.add_conditional_edges(
    "agent", route_after_agent, {"tools": "tools", "finalize": "finalize"}
)
builder.add_edge("tools", "agent")
builder.add_edge("finalize", END)

graph = builder.compile(checkpointer=checkpointer, store=store)


# ---------------------------------------------------------------------------
# 7. Example run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = {
        "configurable": {
            "thread_id": f"session-{uuid.uuid4()}",
            "user_id": "u123",
        }
    }

    user_request = "Plan my day for today, 2026-07-06."
    result = graph.invoke(
        {"messages": [{"role": "user", "content": user_request}]}, config
    )

    while "__interrupt__" in result:
        question = result["__interrupt__"][0].value["question"]
        user_answer = input(question + " ")
        result = graph.invoke(Command(resume=user_answer), config)

    import json
    print(json.dumps(result["output"], indent=2))
