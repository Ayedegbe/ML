import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()                               # pick up OPENAI_API_KEY if you use .env
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def generate_response(
    user_input: str,
    context_documents: list[str],
    model: str = "gpt-4o-mini"              # or any GPT‑4/3.5 model you have access to
) -> str:
    context_text  = "\n\n".join(context_documents[:10])   # keep prompt short
    system_prompt = f"""
                You are TechCorp’s IT Help‑Desk Assistant.

                <CONTEXT>
                {context_text}
                </CONTEXT>

                Rules
                1. Use **only** the information inside <CONTEXT>. If the answer isn't there, apologise and suggest escalation.
                2. Identify the best‑fit issue category (password_reset, wifi_connection, etc.).
                3. Provide a concise, friendly answer:
                    • Numbered or bulleted steps when possible  
                    • Mention the escalation trigger and contact if escalation is needed
                4. Format exactly like:
                Category: <category>

                Response:
                <answer>

                Escalation Required: <Yes/No> """

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_input}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content


