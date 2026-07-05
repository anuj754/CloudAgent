========================================================================
    CLOUD AGENT & DEVICE AGENT SYSTEM - INSTRUCTIONS AND RUNNING STEPS
========================================================================

This guide describes how to configure, start, and test the Cloud and Device 
Agent bidirectional communication system. 

------------------------------------------------------------------------
1. PREREQUISITES
------------------------------------------------------------------------
* Python 3.9+ installed and added to your environmental PATH.

------------------------------------------------------------------------
2. INITIAL ENVIRONMENT SETUP & INSTALLATION
------------------------------------------------------------------------
If the virtual environment is not yet created or dependencies are missing:

A. Create the virtual environment:
   python -m venv venv

B. Activate the virtual environment:
   * Windows (PowerShell):
     .\venv\Scripts\Activate.ps1
   * Windows (Command Prompt):
     .\venv\Scripts\activate.bat
   * Linux/macOS:
     source venv/bin/activate

C. Install all dependencies:
   pip install -r requirements.txt

------------------------------------------------------------------------
3. OPTIONAL ENVIRONMENT CONFIGURATION (.env)
------------------------------------------------------------------------
You can configure LLM settings or device security tokens by creating a `.env` 
file in the root workspace directory:

------------------------------
LLM_PROVIDER=mock
DEVICE_AUTH_TOKEN=super-secret-device-token
OPENAI_API_KEY=your-openai-api-key-if-using-openai
GEMINI_API_KEY=your-gemini-api-key-if-using-gemini
------------------------------

Note: If no API key is specified and LLM_PROVIDER is set to "mock" (default), 
the system will auto-route commands using a rule-based matching simulation.

------------------------------------------------------------------------
4. HOW TO RUN THE SYSTEM (STEP-BY-STEP)
------------------------------------------------------------------------
To run the setup, you need to open two or three separate terminal windows 
with your virtual environment activated:

STEP 1: Start the Cloud Agent Server (Terminal 1)
--------------------------------------------------
Run the following command to start the FastAPI server:
   python run_cloud.py

Wait until you see:
   INFO:     Application startup complete.
   Uvicorn running on http://0.0.0.0:8000

STEP 2: Start the Device Agent (Terminal 2)
--------------------------------------------
In a second terminal window, run the simulated local device agent:
   python mock_device.py

Wait until you see:
   device: Connected to Cloud Agent. Waiting for command dispatches...

STEP 3: Trigger a command execution test (Terminal 3)
------------------------------------------------------
To send a chat prompt requesting a command execution, execute the client:
   python test_client.py

The client will post a request to http://127.0.0.1:8000/api/chat. 
The Cloud Agent triggers LangGraph, intercepts the command "dir", schedules 
it on the connected Device Agent via WebSockets, waits for the command output, 
and returns the final JSON output directly to the test client.
