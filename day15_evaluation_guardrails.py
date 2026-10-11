
import os
import re
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = "gemini-3.5-flash-lite"

KNOWLEDGE_BASE = [
    "Northstar University was founded in 2010.",
    "Northstar University offers Computer Science and Mathematics.",
    "The library closes at 8 PM.",
    "Students can borrow up to 5 books.",
    "The computer lab opens at 9 AM.",
    "Northstar University is located in the fictional city of Pineville."
]

STOP_WORDS = {
    "what", "when", "where", "who", "which", "how", "many", "much",
    "is", "are", "was", "were", "the", "a", "an", "of", "in", "on",
    "at", "to", "does", "do", "did", "can", "i", "you", "it", "its",
    "and", "for", "me", "please", "tell", "about", "university",
    "northstar", "universitys"
}

TESTS = [
    {"question": "When was Northstar University founded?", "expected": "2010", "supported": True},
    {"question": "Which subjects does Northstar University offer?", "expected": "Computer Science", "supported": True},
    {"question": "What time does the library close?", "expected": "8 PM", "supported": True},
    {"question": "Who is the president of Northstar University?", "expected": "I don't have enough information", "supported": False},
    {"question": "What is Northstar University's motto?", "expected": "I don't have enough information", "supported": False}
]

def ask_ai(prompt):
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )
    return response.text or ""

def retrieve_context(question):
    words = set(re.findall(r"[a-z0-9]+", question.lower()))
    words = {word for word in words if word not in STOP_WORDS}
    matches = []
    for fact in KNOWLEDGE_BASE:
        fact_words = set(re.findall(r"[a-z0-9]+", fact.lower()))
        fact_words = {word for word in fact_words if word not in STOP_WORDS}
        score = len(words & fact_words)
        if score > 0:
            matches.append((score, fact))
    matches.sort(key=lambda item: item[0], reverse=True)
    if not matches:
        return []
    return [matches[0][1]]

def guarded_answer(question):
    context = retrieve_context(question)
    if not context:
        return "I don't have enough information to answer that."
    prompt = f"""
Answer using ONLY the facts below.
Do not guess or add outside information.
If the facts do not contain the answer, say:
I don't have enough information to answer that.

Facts:
{chr(10).join(context)}

Question: {question}
Answer:
"""
    return ask_ai(prompt)

def evaluate(answer, test):
    if not test["supported"]:
        return "PASS" if "don't have enough information" in answer.lower() else "FAIL"
    return "PASS" if test["expected"].lower() in answer.lower() else "CHECK"

def main():
    print("DAY 15: AI EVALUATION AND GUARDRAILS")
    results = []
    for index, test in enumerate(TESTS, 1):
        question = test["question"]
        print(f"\nTest {index}: {question}")
        before = ask_ai(
            f"Answer this question directly. If you are unsure, try your best.\nQuestion: {question}"
        )
        after = guarded_answer(question)
        result = evaluate(after, test)
        results.append(result)
        print("BEFORE:", before)
        print("EXPECTED:", test["expected"])
        print("AFTER:", after)
        print("EVALUATION:", result)
    passed = results.count("PASS")
    print(f"\nEvaluation summary: {passed}/{len(TESTS)} passed")
    print("Review every CHECK or FAIL manually.")

if __name__ == "__main__":
    main()
