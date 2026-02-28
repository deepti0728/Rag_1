import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_answer(context, query):
    prompt = f"""
You are a document-grounded AI assistant.

Use ONLY the provided context to answer clearly and concisely.
Do not add facts that are missing from the context.
If the answer is not in the context, say "Information not available in the provided documents."

Context:
{context}

Question:
{query}

Answer:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content


