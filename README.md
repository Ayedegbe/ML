# TechCorp Help-Desk AI API

## Overview
This project is an AI-powered help-desk API for TechCorp, designed to answer IT support questions using a knowledge base and a language model (LLM). It supports escalation logic, provides transparent source references, and can be validated against test scenarios.

## Features
- FastAPI server with `/chat` endpoint for help-desk queries
- Uses OpenAI LLM for response generation, limited to provided context
- Returns answer, category, escalation info, and source document IDs
- `/run_tests` endpoint to validate against test scenarios in `test_requests.json`
- Modular code for retrieval, response generation, and knowledge base management
- Unit tests for main logic
- Secure API key handling via `.env`

## Setup
1. **Clone the repository:**
   ```sh
   git clone https://github.com/Ayedegbe/ML.git
   cd ML
   ```
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
3. **Set up environment variables:**
   - Create a `.env` file in the project root:
     ```
     OPENAI_API_KEY=your-openai-key-here
     ```
4. **Prepare the knowledge base:**
   - Ensure all files in the `knowledge/` folder are present.
   - Run `retriever.py` to build the vector store:
     ```sh
     python retriever.py
     ```

## Running the API Server
```sh
python Api_server.py
```
- The server will start at `http://localhost:8000`.
- Interactive docs available at `/docs`.

## Endpoints
### `/chat` (POST)
- Request:
  ```json
  {
    "question": "How do I reset my password?",
    "top_k": 5
  }
  ```
- Optional query param: `format=html` for HTML output
- Response:
  ```json
  {
    "answer": "...",
    "sources": ["...", "..."]
  }
  ```

### `/run_tests` (GET)
- Runs all scenarios in `test_requests.json` and returns results.

## Testing
- Run unit tests:
  ```sh
  pytest
  "python -m pytest test.py" in powershell
  ```
- Run test scenarios:
  - Use `/run_tests` endpoint or run `run_test_requests.py`.

## Project Structure
- `Api_server.py` — FastAPI server
- `responder.py` — LLM response logic
- `query.py` — Knowledge retrieval
- `retriever.py` — Knowledge base loader/chunker
- `test.py` — Unit tests
- `run_test_requests.py` — Test scenario runner
- `knowledge/` — Markdown/JSON knowledge base
- `.env` — API keys (not tracked)
- `.gitignore` — Excludes cache, data, secrets
## Error Handling

Right now, the API does basic error handling—if you send a bad request or something’s missing, you’ll usually get a clear error message back. If the knowledge base is missing info or the model can’t answer, the responder tries to let you know instead of guessing. There’s room to make this even better, like catching more edge cases and making sure all errors are handled gracefully, but the basics are covered for now.

## Notes
- Do not commit `.env` or any sensitive keys.

## Evaluation

### Accuracy Score: **72%**

#### How’s it doing?
After running a bunch of test questions (using `test_requests.json` and `sample_conversations.json`), the responder gets things right about 72% of the time. It usually picks the right category, follows the format, and knows when to escalate. Most of the mistakes happen when the knowledge base doesn’t have enough info, not because of a problem with the code or logic.

#### What’s working well?
- Answers are clear and easy to follow.
- The format is consistent, so you know what to expect.
- Escalation works as intended in most cases.

#### What could be better?
- The knowledge base could use more details—adding more scenarios, URLs, and contacts would help the responder be even more accurate.
- The prompt instructions can be tweaked a bit more to cut down on those rare “hallucinated” details.
- It’d be good to make sure the responder always says it can’t help if info is missing, instead of guessing.
- More test cases (especially tricky or weird ones) would help catch edge cases.

**How we got this score:**  
We ran a set of sample questions through the responder and checked if the answers stuck to the info in the knowledge base, followed the rules, and handled escalation right. Most of the points lost were just because the knowledge base didn’t have all the answers yet.