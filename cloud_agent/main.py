import logging
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any

from cloud_agent.config import settings
from cloud_agent.connection import manager
from cloud_agent.agent import compiled_graph

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("cloud_agent.main")

app = FastAPI(
    title="Cloud Agent Server",
    description="LangGraph orchestrator managing remote Linux device agents over WebSockets"
)

# CORS middleware for testing in browser or dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Schemas
class MessageItem(BaseModel):
    role: str
    content: str

class ChatPayload(BaseModel):
    messages: List[MessageItem]

@app.get("/health")
def health_check():
    """Confirms service status and lists connected device agents."""
    return {
        "status": "healthy",
        "llm_provider": settings.LLM_PROVIDER,
        "connected_devices": list(manager.active_devices.keys())
    }

# ----------------- DEVICE WORKSPACE ENDPOINT -----------------

@app.websocket("/ws/device/{device_id}")
async def device_websocket_endpoint(
    websocket: WebSocket, 
    device_id: str,
    token: str = Query(...)
):
    """
    WebSocket endpoint for Device Agents to register and maintain connection.
    Secured via query parameter token auth.
    """
    if token != settings.DEVICE_AUTH_TOKEN:
        logger.warning(f"Unauthorized connection attempt from device {device_id}.")
        await websocket.close(code=4003, reason="Invalid credentials")
        return

    await manager.connect_device(device_id, websocket)

    try:
        while True:
            # Continuously listen to event payloads from the device (e.g., updates, completions)
            data = await websocket.receive_json()
            logger.debug(f"Received from device '{device_id}': {data}")
            await manager.handle_device_message(device_id, data)
    except WebSocketDisconnect:
        manager.disconnect_device(device_id)
    except Exception as e:
        logger.error(f"Error handling device connection '{device_id}': {e}")
        manager.disconnect_device(device_id)


# ----------------- USER ACCESS REST API -----------------

@app.post("/api/chat")
async def chat_interaction(payload: ChatPayload = Body(...)):
    """
    Synchronous REST API for sending prompts to the Cloud Agent.
    Runs the LangGraph, waiting for execution (including device commands) to resolve.
    """
    # Map input messages to LangGraph schema
    graph_messages = []
    for msg in payload.messages:
        graph_messages.append((msg.role, msg.content))

    if not graph_messages:
        raise HTTPException(status_code=400, detail="Messages list cannot be empty.")

    # Simple console progress logger callback if executing command
    async def console_progress(data: dict):
        log_line = data.get("log", "")
        logger.info(f"Progress updating: {log_line}")

    config = {
        "configurable": {
            "progress_callback": console_progress
        }
    }

    try:
        # Run graph in async mode
        state = {"messages": graph_messages}
        result = await compiled_graph.ainvoke(state, config=config)
        
        # Get final agent response
        final_message = result["messages"][-1]
        
        return {
            "response": final_message.content,
            "all_messages": [
                {"role": "user" if getattr(m, "type", "user") == "human" else "assistant", "content": m.content}
                for m in result["messages"]
            ]
        }
    except Exception as e:
        logger.error(f"Error during agent chat invocation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ----------------- USER STREAMING WORKSPACE ENDPOINT -----------------

@app.websocket("/ws/chat")
async def user_chat_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for User clients to receive real-time token/log updates
    concerning agent thoughts and active device execution logs.
    """
    await websocket.accept()
    logger.info("User client connected to Chat WebSocket.")

    try:
        while True:
            # Wait for user query
            data = await websocket.receive_json()
            query = data.get("query")
            if not query:
                await websocket.send_json({"type": "error", "message": "query field is missing"})
                continue

            logger.info(f"User WebSocket Chat Query: {query}")

            # Define localized progress callback mapping this request's device logs
            # directly back to this user's socket
            async def pipe_progress_to_user(progress_payload: dict):
                try:
                    await websocket.send_json({
                        "type": "progress",
                        "log": progress_payload.get("log", ""),
                        "request_id": progress_payload.get("request_id")
                    })
                except Exception as ex:
                    logger.error(f"Failed to pipe progress to user: {ex}")

            config = {
                "configurable": {
                    "progress_callback": pipe_progress_to_user
                }
            }

            # Notify user agent is thinking
            await websocket.send_json({"type": "status", "message": "LangGraph invoked decision routing..."})

            # Run LangGraph Agent
            try:
                state = {"messages": [("user", query)]}
                result = await compiled_graph.ainvoke(state, config=config)
                
                final_response = result["messages"][-1].content
                await websocket.send_json({
                    "type": "result",
                    "response": final_response
                })
            except Exception as graph_err:
                logger.error(f"LangGraph execution error: {graph_err}")
                await websocket.send_json({
                    "type": "error",
                    "message": f"Execution error: {str(graph_err)}"
                })

    except WebSocketDisconnect:
        logger.info("User client disconnected from Chat WebSocket.")
    except Exception as e:
        logger.error(f"Error on user WebSocket channel: {e}")
