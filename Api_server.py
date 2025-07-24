
# Import FastAPI and supporting libraries
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
# Import core logic modules
from query import query_helpdesk
from responder import generate_response
import uvicorn
# Main entry point: Run the API server
import webbrowser
import threading  
import re


# Initialize FastAPI app
app = FastAPI(title="TechCorp Help‑Desk API")


# Request model for /chat endpoint
class ChatRequest(BaseModel):
    question: str  # User's helpdesk question
    top_k: int = 5  # Number of top results to retrieve (optional)


# Response model for /chat endpoint
class ChatResponse(BaseModel):
    answer: str  # LLM-generated answer
    sources: list[str]  # Source document IDs


# /chat endpoint: Handles help-desk queries
# Accepts question and top_k, returns answer and sources
# Optional query param 'format' for HTML/text output
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    # Validate input
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Retrieve relevant knowledge chunks
    results = query_helpdesk(req.question, top_k=req.top_k)
    context_chunks = [doc for doc, _meta in results]
    # Generate response using LLM
    answer_text = generate_response(req.question, context_chunks)

    # Get format query param (default to 'text')
    format_type = request.query_params.get('format', 'text')
    if format_type == 'html':
      
        # Replace double newlines with paragraph breaks
        answer_out = re.sub(r'\n\n+', '<br><br>', answer_text)
        # Replace single newlines with line breaks
        answer_out = re.sub(r'(?<!<br>)\n', '<br>', answer_out)
    else:
        answer_out = answer_text

    # Collect source document IDs for transparency
    source_ids = [meta["parent_id"] for _doc, meta in results]
    return ChatResponse(answer=answer_out, sources=source_ids)



# /run_tests endpoint: Runs all test scenarios from test_requests.json
# Returns results for each scenario (question, expected, answer)
@app.get("/run_tests")
def run_test_scenarios():
    import json
    from pathlib import Path
    import os
    base_dir = Path(__file__).parent
    test_file = base_dir / "test_requests.json"
    if not test_file.exists():
        return {"error": "test_requests.json not found"}
    data = json.loads(test_file.read_text(encoding="utf-8"))
    test_requests = data.get("test_requests", [])
    results = []
    for req in test_requests:
        question = req["request"]
        expected = req.get("expected_classification")
        elements = req.get("expected_elements")
        escalate = req.get("escalate")
        api_req = ChatRequest(question=question, top_k=5)
        # Use same logic as /chat endpoint
        chunks = [doc for doc, _meta in query_helpdesk(api_req.question, top_k=api_req.top_k)]
        answer = generate_response(api_req.question, chunks)
        # Only include the question and relevant expected/answer fields
        results.append({
            "question": question, 
            "answer": answer
        })
    return {"test_results": results}




# Main entry point: Run the API server and open browser
if __name__ == "__main__":
    def open_browser():
        webbrowser.open_new("http://localhost:8000/docs")
    threading.Timer(1.5, open_browser).start()
    uvicorn.run("Api_server:app", host="0.0.0.0", port=8000, reload=True)
