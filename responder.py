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
            f"- {cat}: {details['description']}\n  Key elements: {', '.join(details.get('key_elements', []))}\n  Escalation triggers: {', '.join(details.get('escalation_triggers', []))}" for cat, details in categories.items()
        ])
        categories_section = (
            "Available categories (use the best match):\n" +
            categories_list +
            "\n\nFor the selected category, your answer must address all key elements listed.\nEscalate ONLY if the user's issue matches a listed escalation trigger for the chosen category."
        )
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
                1. You MUST use only the information provided inside <CONTEXT>. Do NOT use any outside knowledge, general IT advice, or steps not found in the context.
                2. If the answer is not present in the context, apologize and suggest escalation.
                3. If the context contains a website, email address, or procedure, you MUST use it exactly as written. Do NOT invent or substitute URLs, emails, or steps.
                4. If the context contains a specific procedure or step, follow it exactly.
                5. Quote or paraphrase directly from the context whenever possible, but you may rephrase for clarity and user-friendliness.
                6. Do NOT invent, generalize, or add any steps or advice not found verbatim in the context.
                7. Identify the best-fit issue category from the list above.
                8. Your answer must mention or address all key elements for the selected category.
                9. Provide a clear, complete, and friendly answer:
                    • Use numbered or bulleted steps when possible
                    • Use full sentences and natural language, not just keywords
                    • Mention the escalation trigger and contact ONLY if escalation is required.
                10. Escalate ONLY if the user's issue matches a listed escalation trigger for the chosen category.
                11. Format your output exactly like this:
                Category: <category>

                Response:
                <answer>

                Escalation Required: <Yes/No> (Only say Yes if the user's issue matches an escalation trigger for the selected category.)
                """
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


