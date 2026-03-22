# =============================================================================
# 03_client.py
# =============================================================================
# This is the client that bootcamp participants use to query the RAG API.
# It runs as an interactive terminal session where you type questions and
# get answers back from the API server.
#
# Make sure the API server is running first:
#   uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
#
# Then run this client:
#   python 03_client.py
# =============================================================================
 
import requests
import json
import os
from dotenv import load_dotenv
 
# Load .env so we can read the server URL if set there
load_dotenv()
 
# =============================================================================
# CONFIGURATION
# =============================================================================
 
# The URL of the running API server
# Change this if the server is running on a different host or port
API_URL = os.getenv("API_URL", "http://localhost:8000")
 
# How many source chunks to show below each answer
# Set to 0 to hide source chunks entirely
SHOW_SOURCES = True
MAX_SOURCES_TO_SHOW = 200
 
 
# =============================================================================
# HELPER: PRINT A DIVIDER LINE
# =============================================================================
 
def divider(char="─", width=60):
    print(char * width)
 
 
# =============================================================================
# HELPER: CHECK THE SERVER IS REACHABLE
# =============================================================================
 
def check_server():
    """
    Pings the /health endpoint before starting the chat loop.
    Exits with a helpful message if the server isn't running.
    """
    try:
        response = requests.get(f"{API_URL}/health", timeout=30)
        data = response.json()
 
        print(f"  ✅ Server is online")
        print(f"  📦 Chunks in database : {data.get('chunks_stored', '?')}")
        print(f"  🤖 Model              : {data.get('llm_model', '?')}")
        print(f"  🔑 API key set        : {data.get('api_key_set', False)}")
 
        if not data.get("api_key_set"):
            print("\n  ⚠️  WARNING: The server has no API key set.")
            print("     Queries will fail until OPENROUTER_API_KEY is set on the server.")
 
        return True
 
    except requests.exceptions.ConnectionError:
        print(f"\n  ❌ Could not connect to the server at {API_URL}")
        print("  Make sure the server is running with:")
        print("  uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload")
        return False
 
    except Exception as e:
        print(f"\n  ❌ Unexpected error checking server: {e}")
        return False
 
 
# =============================================================================
# HELPER: SEND A QUESTION TO THE API
# =============================================================================
 
def ask_question(question: str) -> dict | None:
    """
    Sends a question to the POST /query endpoint and returns the response dict.
    Returns None if the request fails.
    """
    try:
        # Build the request payload
        payload = {"question": question}
 
        # Send the POST request to the API
        response = requests.post(
            f"{API_URL}/query",
            json=payload,         # automatically sets Content-Type: application/json
            timeout=60            # wait up to 60 seconds for the LLM to respond
        )
 
        # Check for HTTP errors (4xx, 5xx)
        response.raise_for_status()
 
        # Parse and return the JSON response
        return response.json()
 
    except requests.exceptions.Timeout:
        print("\n  ❌ Request timed out. The LLM is taking too long to respond.")
        print("     Try again or check if OpenRouter is having issues.")
        return None
 
    except requests.exceptions.HTTPError as e:
        print(f"\n  ❌ Server returned an error: {e}")
        # Try to show the server's error message if there is one
        try:
            error_detail = response.json().get("detail", "No details provided")
            print(f"     Detail: {error_detail}")
        except Exception:
            pass
        return None
 
    except Exception as e:
        print(f"\n  ❌ Unexpected error: {e}")
        return None
 
 
# =============================================================================
# HELPER: PRINT THE RESPONSE NICELY
# =============================================================================
 
def print_response(data: dict):
    """
    Formats and prints the API response in a readable way.
    Shows the answer first, then the source chunks below it.
    """
 
    # Print the LLM's answer
    print("\n🤖 Answer:")
    divider()
    print(data.get("answer", "No answer returned"))
    divider()
 
    # Optionally print the source chunks so users can see where the answer came from
    if SHOW_SOURCES:
        context = data.get("context", [])
 
        if context:
            print(f"\n📚 Sources ({min(MAX_SOURCES_TO_SHOW, len(context))} of {len(context)} chunks shown):")
 
            # Only show up to MAX_SOURCES_TO_SHOW chunks to keep output clean
            for i, chunk in enumerate(context[:MAX_SOURCES_TO_SHOW]):
                print(f"\n  Source {i + 1}: {chunk.get('source', 'unknown')}")
 
                # Show page number if available (PDF chunks have this)
                page = chunk.get("metadata", {}).get("page_label")
                if page:
                    print(f"  Page   : {page}")
 
                # Show DTC code if this chunk came from the CSV
                dtc_code = chunk.get("metadata", {}).get("code")
                if dtc_code:
                    print(f"  DTC    : {dtc_code}")
 
                # Show a short preview of the chunk text
                preview = chunk.get("content", "")[:200].replace("\n", " ")
                print(f"  Preview: {preview}...")
 
    print(f"\n  Model: {data.get('model_used', 'unknown')}")
 
 
# =============================================================================
# MAIN - Interactive chat loop
# =============================================================================
 
if __name__ == "__main__":
 
    # --- Welcome banner ---
    print("\n")
    divider("═")
    print("  🔧 1.4L TSI Engine Assistant — RAG Client")
    print("  Based on VW SSP 359 + DTC Code Database")
    divider("═")
 
    # --- Check the server is up before starting ---
    print("\n🔌 Connecting to API server...")
    if not check_server():
        exit(1)
 
    # --- Instructions ---
    print("\n  Type your question and press Enter.")
    print("  Commands:")
    print("    'sources on'  — show source chunks (default)")
    print("    'sources off' — hide source chunks")
    print("    'quit' or 'exit' — close the client")
    divider()
 
    # --- Interactive question loop ---
    while True:
        try:
            # Get input from the user
            print()
            question = input("❓ Your question: ").strip()
 
            # Skip empty input
            if not question:
                continue
 
            # Handle exit commands
            if question.lower() in ("quit", "exit", "q"):
                print("\n👋 Goodbye!\n")
                break
 
            # Handle toggle commands for source display
            if question.lower() == "sources off":
                SHOW_SOURCES = False
                print("   ✅ Source chunks hidden")
                continue
 
            if question.lower() == "sources on":
                SHOW_SOURCES = True
                print("   ✅ Source chunks visible")
                continue
 
            # Send the question to the API
            print("\n⏳ Searching knowledge base and generating answer...")
            result = ask_question(question)
 
            # Print the response if we got one
            if result:
                print_response(result)
 
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print("\n\n👋 Interrupted. Goodbye!\n")
            break