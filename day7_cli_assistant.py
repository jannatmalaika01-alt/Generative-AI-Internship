from google import genai
from google.genai import types

# Set your key first (never put it in this file or in screenshots):
# PowerShell:  $env:GEMINI_API_KEY="your-key-here"
client = genai.Client()

MODEL = "gemini-3.5-flash-lite"

config = types.GenerateContentConfig(
    system_instruction="You are a concise study assistant.",
    thinking_config=types.ThinkingConfig(thinking_level="low"),
    max_output_tokens=500,
)

print("AI Assistant (type 'quit' to exit)")

while True:
    user_input = input("\nYou: ").strip()
    if user_input.lower() in ("quit", "exit"):
        print("Goodbye!")
        break
    if not user_input:
        continue

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=user_input,
            config=config,
        )
        print(f"\nAI: {response.text}")

        usage = response.usage_metadata
        print("\n--- Token usage ---")
        print(f"Prompt tokens: {usage.prompt_token_count}")
        print(f"Response tokens: {usage.candidates_token_count}")
    except Exception as e:
        print(f"\nRequest failed: {e}")