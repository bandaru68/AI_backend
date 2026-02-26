import os
from openai import OpenAI

# Create a client object to talk to OpenAI's API
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def build_rag_prompt(question: str, context: None) -> str:
    # if not context:
    #     context = "No external knowledge was retrieved. Answer based only on your general knowledge."

    prompt = f"""
You are an AI assistant in a Retrieval-Augmented Generation (RAG) style system.

Context:
{context}

User question:
{question}

Instructions:
- Be clear and concise.
- If the answer is not obvious, say that you are not completely sure.
"""
    return prompt.strip()

def generate_answer_rag_style(question: str, context: str | None = None) -> str:
    """
    Generation part of RAG:
    1. Build the prompt (with context if we had retrieval).
    2. Call OpenAI to get an answer.
    """
    # Step 1: create the full prompt string
    prompt = build_rag_prompt(question, context)
    # prompt = "What is capital of India?"
    # Step 2: send the prompt to OpenAI's chat completion endpoint
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1000,
        temperature=0.8,
    )

    # Step 3: extract just the text of the answer from the response
    return response.choices[0].message.content

if __name__ == "__main__":
    # This is the question we want to ask in RAG-style
    user_question = "Explain what a vector database is in a RAG system"

    # For now we are not doing retrieval, so context is None
    answer = generate_answer_rag_style(user_question, context=None)

    print("Question:", user_question)
    print("Answer:", answer)