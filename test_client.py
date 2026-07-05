import httpx
import json

URL = "http://127.0.0.1:8000/api/chat"

payload = {
    "messages": [
        {
            "role": "user",
            "content": "Hi cloud agent, please execute the command 'dir' on the device."
        }
    ]
}

print(f"Sending POST request to: {URL}")
print(f"Payload: {json.dumps(payload, indent=2)}")

try:
    response = httpx.post(URL, json=payload, timeout=30.0)
    print(f"\nResponse status code: {response.status_code}")
    print("Response JSON:")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Error calling server: {e}")
