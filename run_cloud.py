import uvicorn
from cloud_agent.config import settings

if __name__ == "__main__":
    print(f"Starting Cloud Agent server on {settings.HOST}:{settings.PORT}")
    print(f"LLM Provider loaded: {settings.LLM_PROVIDER}")
    print(f"WS endpoint: ws://127.0.0.1:{settings.PORT}/ws/device/<device_id>?token={settings.DEVICE_AUTH_TOKEN}")
    
    uvicorn.run(
        "cloud_agent.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True
    )
