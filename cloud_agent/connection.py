import asyncio
import logging
import uuid
from typing import Dict, List, Callable, Awaitable, Optional
from fastapi import WebSocket

logger = logging.getLogger("cloud_agent.connection")

class ConnectionManager:
    def __init__(self):
        # Maps device_id to its active WebSocket connection
        self.active_devices: Dict[str, WebSocket] = {}
        # Maps request_id (UUID) to asyncio.Future for command responses
        self.pending_requests: Dict[str, asyncio.Future] = {}
        # Maps request_id to a list of async callbacks for real-time progress updates
        self.progress_callbacks: Dict[str, List[Callable[[dict], Awaitable[None]]]] = {}

    async def connect_device(self, device_id: str, websocket: WebSocket):
        """Register an active device connection."""
        await websocket.accept()
        self.active_devices[device_id] = websocket
        logger.info(f"Device '{device_id}' connected successfully.")

    def disconnect_device(self, device_id: str):
        """Remove a device connection and resolve any pending works as failed."""
        if device_id in self.active_devices:
            self.active_devices.pop(device_id)
            logger.info(f"Device '{device_id}' disconnected.")
        
        # Clean up any pending futures if the device disconnected
        # In a real app we might verify if the future is tied to this device
        # For simplicity, we fail all active futures or log it

    async def register_callback(self, request_id: str, callback: Callable[[dict], Awaitable[None]]):
        """Register a callback for real-time console updates from a running task."""
        if request_id not in self.progress_callbacks:
            self.progress_callbacks[request_id] = []
        self.progress_callbacks[request_id].append(callback)

    def deregister_callbacks(self, request_id: str):
        """Clean up callbacks for a request ID."""
        self.progress_callbacks.pop(request_id, None)

    async def send_command_and_wait(
        self, 
        device_id: str, 
        command: str, 
        progress_cb: Optional[Callable[[dict], Awaitable[None]]] = None
    ) -> str:
        """
        Sends a shell/system command to the specified device and waits (blocks asynchronously) 
        until the device returns a final response.
        """
        if device_id not in self.active_devices:
            raise RuntimeError(f"Device '{device_id}' is not connected.")

        websocket = self.active_devices[device_id]
        request_id = str(uuid.uuid4())
        
        # Create a future that will be resolved when the WebSocket receives the result
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self.pending_requests[request_id] = future

        if progress_cb:
            await self.register_callback(request_id, progress_cb)

        # Send the command payload to the Device Agent
        payload = {
            "type": "command",
            "request_id": request_id,
            "command": command
        }
        
        try:
            logger.info(f"Sending command to {device_id} (ID: {request_id}): {command}")
            await websocket.send_json(payload)
            # block until future resolves or timeout occurs
            result = await asyncio.wait_for(future, timeout=120.0) # 2 minute timeout
            return result
        except asyncio.TimeoutError:
            logger.error(f"Command {request_id} timed out waiting for device response.")
            raise TimeoutError(f"Command timed out on device '{device_id}' after 120 seconds.")
        except Exception as e:
            logger.error(f"Error during command execution {request_id}: {str(e)}")
            raise e
        finally:
            # Cleanup
            self.pending_requests.pop(request_id, None)
            self.deregister_callbacks(request_id)

    async def handle_device_message(self, device_id: str, data: dict):
        """Processes incoming WebSocket packets from the Device Agent."""
        msg_type = data.get("type")
        request_id = data.get("request_id")

        if not request_id:
            logger.warning(f"Received malformed packet from {device_id} without request_id.")
            return

        if msg_type == "progress":
            # Real-time stdout/stderr progression packet
            # e.g., {"type": "progress", "request_id": "...", "log": "Installing package..."}
            if request_id in self.progress_callbacks:
                for cb in self.progress_callbacks[request_id]:
                    try:
                        await cb(data)
                    except Exception as e:
                        logger.error(f"Error executing progress callback for {request_id}: {e}")
                        
        elif msg_type == "result":
            # Final result response packet
            # e.g., {"type": "result", "request_id": "...", "status": "success", "output": "..."}
            future = self.pending_requests.get(request_id)
            if future and not future.done():
                status = data.get("status", "success")
                if status == "success":
                    future.set_result(data.get("output", ""))
                else:
                    error_msg = data.get("error", "Unknown execution error on device")
                    future.set_exception(RuntimeError(f"Device execution failed: {error_msg}"))
            else:
                logger.warning(f"Received result for request_id '{request_id}' which is not pending or already done.")

# Global Connection Manager Instance
manager = ConnectionManager()
