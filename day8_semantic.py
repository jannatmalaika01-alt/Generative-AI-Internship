import math
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = "gemini-embedding-001"

def embed(texts, task_type):
    """Send a list of texts to the embeddings API and return one vector per text."""
    response = client.models.embed_content(
        model=MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return [e.values for e in response.embeddings]

def cosine(a, b):
    """Cosine similarity: closer to 1 means more similar in meaning."""
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))

originals = [
    "I enjoy learning Python programming.",
    "Our school offers evening classes for adult learners.",
    "Pasta with tomato sauce is my favorite dinner.",
    "The football team won the championship last night.",
    "We are planning a trip to the mountains this summer.",
]
rewrites = [
    "I like studying software development with Python.",
    "Spaghetti topped with a red sauce is what I love to eat for supper.",
    "We will travel to the hills for a vacation soon.",
]

orig_vecs = embed(originals, "SEMANTIC_SIMILARITY")
new_vecs = embed(rewrites, "SEMANTIC_SIMILARITY")

print("Embedding length:", len(orig_vecs[0]))
print("First 5 numbers of sentence 1:", [round(v, 4) for v in orig_vecs[0][:5]])

print("\n=== Experiment 1: related sentences with different words ===")
for text, vec in zip(rewrites, new_vecs):
    scores = [cosine(vec, o) for o in orig_vecs]
    best = max(range(len(scores)), key=lambda i: scores[i])
    print(f"\nNew sentence: {text}")
    print(f"Closest original: {originals[best]}  (score {scores[best]:.3f})")

documents = [
    "BrightPath Online Academy offers beginner to advanced programming courses in Python, JavaScript, and web development.",
    "Students can enroll in any course at any time, and lessons are available on demand around the clock.",
    "A monthly subscription costs $20 and gives access to all courses, with a 7-day free trial.",
    "Students who finish a course and pass the final project receive a verified certificate.",
    "Refunds are available within 14 days of purchase if the student has completed less than 20 percent of the course.",
    "The support team answers questions by email and live chat from 9 AM to 6 PM on weekdays.",
    "Each course includes quizzes, coding exercises, and a final project reviewed by an instructor.",
    "Students can download lesson videos to watch offline in the mobile app.",
    "Instructors are working software engineers with at least five years of industry experience.",
    "Learners who need a lower price can apply for a scholarship that covers up to 50 percent of the subscription fee.",
]
questions = [
    "How much does it cost to join?",
    "Can I get my money back if I don't like a course?",
    "Will I get proof that I completed a course?",
    "Can I study without an internet connection?",
    "Who do I contact if I have a problem?",
    "Is there financial help for students who can't afford it?",
]

doc_vecs = embed(documents, "RETRIEVAL_DOCUMENT")
q_vecs = embed(questions, "RETRIEVAL_QUERY")

print("\n\n=== Experiment 2: semantic search (top 3 matches per question) ===")
for q, qv in zip(questions, q_vecs):
    ranked = sorted(
        ((cosine(qv, dv), i) for i, dv in enumerate(doc_vecs)), reverse=True
    )[:3]
    print(f"\nQuestion: {q}")
    for rank, (score, i) in enumerate(ranked, start=1):
        print(f"  {rank}. [Info {i + 1}] score {score:.3f}  {documents[i]}")
