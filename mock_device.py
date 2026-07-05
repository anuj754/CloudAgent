import asyncio
import json
import logging
import sys
import websockets

# Setup logger details
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] device: %(message)s"
)
logger = logging.getLogger("device_agent")

CLOUD_URL = "ws://127.0.0.1:8000/ws/device/linux-laptop-01?token=super-secret-device-token"

async def execute_and_stream(command: str, websocket, request_id: str):
    """
    Executes a shell command on the host (e.g. Linux Laptop) 
    and streams stdout/stderr updates line by line via WebSocket.
    """
    logger.info(f"Received Execution Command: {command}")
    
    # Notify server command is starting
    await websocket.send(json.dumps({
        "type": "progress",
        "request_id": request_id,
        "log": f"Starting command execution: '{command}' on host device..."
    }))

    # Async execute subprocess
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
    except Exception as e:
        logger.error(f"Failed to start subprocess: {e}")
        await websocket.send(json.dumps({
            "type": "result",
            "request_id": request_id,
            "status": "error",
            "error": f"Failed to spawn subprocess: {str(e)}"
        }))
        return

    # Accumulator for stdout
    stdout_accumulator = []

    # Read stdout line-by-line and stream it
    async def read_stream(stream, prefix):
        while True:
            line = await stream.readline()
            if not line:
                break
            
            decoded_line = line.decode().rstrip()
            logger.info(f"[{prefix}] {decoded_line}")
            stdout_accumulator.append(decoded_line)
            
            # Send real-time console updates to Cloud Agent
            try:
                await websocket.send(json.dumps({
                    "type": "progress",
                    "request_id": request_id,
                    "log": decoded_line
                }))
            except Exception as wse:
                logger.error(f"Websocket send failed during streaming payload: {wse}")

    # Gather stdout stream reader (we also read stderr streams similarly)
    await asyncio.gather(
        read_stream(proc.stdout, "stdout"),
        read_stream(proc.stderr, "stderr")
    )

    # Wait for process finish
    exit_code = await proc.wait()
    logger.info(f"Command process finished with exit code {exit_code}")

    full_output = "\n".join(stdout_accumulator)

    # Dispatch final response JSON packet
    if exit_code == 0:
        result_payload = {
            "type": "result",
            "request_id": request_id,
            "status": "success",
            "output": full_output
        }
    else:
        result_payload = {
            "type": "result",
            "request_id": request_id,
            "status": "error",
            "error": f"Execution returned exit-code {exit_code}",
            "output": full_output
        }
    
    await websocket.send(json.dumps(result_payload))
    logger.info(f"Command completion status sent.")

async def run_device_client():
    logger.info(f"Connecting to Cloud Agent WebSocket: {CLOUD_URL}")
    while True:
        try:
            async with websockets.connect(CLOUD_URL) as websocket:
                logger.info("Connected to Cloud Agent. Waiting for command dispatches...")
                async for message in websocket:
                    data = json.loads(message)
                    
                    if data.get("type") == "command":
                        cmd = data.get("command")
                        req_id = data.get("request_id")
                        
                        # run asynchronously without blocking the socket read loop
                        asyncio.create_task(execute_and_stream(cmd, websocket, req_id))
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Connection closed by Cloud Agent. Reconnecting in 5 seconds...")
            await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"Websocket connection error: {e}. Reconnecting in 5 seconds...")
            await asyncio.sleep(5)

if __name__ == "__main__":
    try:
        asyncio.run(run_device_client())
    except KeyboardInterrupt:
        logger.info("Device Agent terminated by user.")
        sys.exit(0)
