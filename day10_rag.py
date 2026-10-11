import chromadb
from google import genai
from google.genai import types
from pypdf import PdfReader

# The API key is read from the GEMINI_API_KEY environment variable.
client_genai = genai.Client()
client_chroma = chromadb.PersistentClient(path="./study_buddy_db")
collection = client_chroma.get_or_create_collection(name="real_notes")

EMBED_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-3.5-flash-lite"
PDF_PATH = "web servlets.pdf"  # your PDF, in the same folder as this script


def get_embeddings(texts, task_type):
    """Embed a list of texts. Sends up to 100 texts per request, which is faster
    and avoids hitting the free-tier rate limit."""
    vectors = []
    for i in range(0, len(texts), 100):
        result = client_genai.models.embed_content(
            model=EMBED_MODEL,
            contents=texts[i:i + 100],
            config=types.EmbedContentConfig(task_type=task_type),
        )
        vectors.extend(e.values for e in result.embeddings)
    return vectors


def extract_text_from_pdf(path):
    reader = PdfReader(path)
    full_text = ""
    for page in reader.pages:
        full_text += (page.extract_text() or "") + "\n"  # "or ''" handles pages with no text
    return full_text


def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into chunks of chunk_size characters, with a bit of overlap
    so we don't cut a sentence in half right at a chunk boundary and lose meaning."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


# --- Step 1-3: read the PDF and split it into chunks ---
raw_text = extract_text_from_pdf(PDF_PATH)
chunks = chunk_text(raw_text)
print(f"Extracted {len(raw_text)} characters, split into {len(chunks)} chunks.")

# --- Step 4-5: embed the chunks and store them in Chroma (only the first time) ---
if collection.count() == 0:
    embeddings = get_embeddings(chunks, "RETRIEVAL_DOCUMENT")
    collection.add(documents=chunks, embeddings=embeddings, ids=[f"chunk_{i}" for i in range(len(chunks))])
    print(f"Stored {len(chunks)} chunks in Chroma.")
else:
    print(f"Collection already has {collection.count()} chunks - skipping re-embedding.")

# --- Step 6-8: ask questions about the PDF ---
while True:
    question = input("\nAsk a question about your notes (or type 'quit'): ").strip()
    if question.lower() == "quit":
        break
    if not question:
        continue

    # Retrieval: find the 3 chunks closest in meaning to the question
    question_embedding = get_embeddings([question], "RETRIEVAL_QUERY")
    results = collection.query(query_embeddings=question_embedding, n_results=3)
    retrieved = results["documents"][0]

    print("\nRetrieved chunks:")
    for i, chunk in enumerate(retrieved, start=1):
        print(f"  {i}. {chunk[:120].replace(chr(10), ' ')}...")

    # Augmentation: put the retrieved chunks and the question into one prompt
    context = "\n".join(f"- {c}" for c in retrieved)
    rag_prompt = f"""Answer using ONLY the context below.
If the answer is not in the context, reply exactly: "I could not find this in your notes."

Context:
{context}

Question: {question}
Answer:"""

    # Generation: the LLM writes the answer from the context
    try:
        response = client_genai.models.generate_content(
            model=CHAT_MODEL,
            contents=rag_prompt,
            config={"system_instruction": "You are a study assistant that only answers from provided notes."},
        )
        print("\nAI:", response.text)
    except Exception as e:
        print("\nRequest failed:", e)