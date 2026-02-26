import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# ---- OpenAI client ----
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# Allow React dev server to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Request schema from React ----
class AskRequest(BaseModel):
    question: str

# ---- Read context ----
def read_context_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# ---- Build prompt ----
def build_rag_prompt(question: str, context: str) -> str:
    prompt = f"""
rt these requirements into a table with columns:
- Requirement
- Why it matters (short)
- Improvement suggestion (if applicable)Conve

Context:
{context}

User question:
{question}

Rules:
- Answer ONLY using the context.
- If the context does NOT contain the answer, reply EXACTLY:
  OUT_OF_CONTEXT
- Do not guess or add external knowledge.
""".strip()
    return prompt.strip()

# ---- OpenAI call ----
def generate_answer(question: str) -> str:
    context = read_context_file("context.txt")
    prompt = build_rag_prompt(question, context)

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are an assistant that may use the provided context when it is relevant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=800,
    )
    return response.choices[0].message.content

# ---- API endpoint ----
@app.post("/ask")
def ask(req: AskRequest):
    answer = generate_answer(req.question)
    return {"answer": answer}