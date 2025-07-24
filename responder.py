import os
from openai import OpenAI  # OpenAI API client
from dotenv import load_dotenv  # For loading .env variables
load_dotenv()  # Load environment variables from .env
api_key = os.getenv("OPENAI_API_KEY")  # Get OpenAI API key
client = OpenAI(api_key=api_key)  # Initialize OpenAI client

def generate_response(
    user_input: str,
    context_documents: list[str],
    model: str = "gpt-4o-mini"  # or any GPT‑4/3.5 model you have access to
) -> str:
    # Join up to 10 context documents for the LLM prompt
    context_text = "\n\n".join(context_documents[:10])

    # Load categories and descriptions for better LLM guidance
    import json
    from pathlib import Path
    categories_path = Path("knowledge/categories.json")
    if categories_path.exists():
        categories_data = json.loads(categories_path.read_text(encoding="utf-8"))
        categories = categories_data.get("categories", {})
        categories_list = "\n".join([
            f"- {cat}: {details['description']}" for cat, details in categories.items()
        ])
        categories_section = f"Available categories (use the best match):\n{categories_list}"
    else:
        categories_section = ""

    # Prompt instructs the LLM to answer only from context and follow help-desk rules
    system_prompt = f"""
                You are TechCorp’s IT Help‑Desk Assistant.

                <CONTEXT>
                {context_text}
                </CONTEXT>

                {categories_section}

                Rules
                1. Use **only** the information inside <CONTEXT> . If the answer isn't there, apologise and suggest escalation.
                2. Identify the best‑fit issue category from the list above.
                3. Provide a concise, friendly answer:
                    • Numbered or bulleted steps when possible  
                    • Mention the escalation trigger and contact if escalation is needed
                4. Format exactly like:
                Category: <category>

                Response:
                <answer>

                Escalation Required: <Yes/No> """
    # Call OpenAI LLM with system and user prompt
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_input}
        ],
        temperature=0.2
    )
    # Return the generated answer
    return response.choices[0].message.content


