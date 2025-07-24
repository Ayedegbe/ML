# Script to run all test requests against the /chat API endpoint
import json
import requests

# URL of the local API server
API_URL = "http://localhost:8000/chat"

# Load test scenarios from test_requests.json
with open("test_requests.json", "r", encoding="utf-8") as f:
    data = json.load(f)
    test_requests = data["test_requests"]

# Loop through each test scenario and send to API
for req in test_requests:
    payload = {
        "question": req["request"],
        "top_k": 5
    }
    # Send POST request to /chat endpoint
    response = requests.post(API_URL, json=payload)
    # Print test scenario and API response
    print(f"Request: {req['request']}")
    print(f"Expected classification: {req['expected_classification']}")
    print(f"Expected elements: {req['expected_elements']}")
    print(f"Escalate: {req['escalate']}")
    print(f"API Response: {response.json()['answer']}")
    print(f"Sources: {response.json()['sources']}")
    print('-' * 80)
