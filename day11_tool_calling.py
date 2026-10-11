from google import genai
from google.genai import types

# The API key is read from the GEMINI_API_KEY environment variable.
client = genai.Client()
MODEL = "gemini-3.5-flash-lite"

# ---------- 1. Fictional order data (no real database needed) ----------
ORDERS = {
    "ORD-101": {"item": "Wireless Keyboard", "status": "Shipped", "expected_delivery": "2026-10-15"},
    "ORD-102": {"item": "USB-C Hub", "status": "Processing", "expected_delivery": "2026-10-20"},
    "ORD-103": {"item": "Laptop Stand", "status": "Delivered", "delivered_on": "2026-10-08"},
}


# ---------- 2. The tool: a normal Python function ----------
def check_order_status(order_id):
    order = ORDERS.get(order_id.strip().upper())
    if order is None:
        return {"error": f"No order found with ID {order_id}"}
    return {"order_id": order_id.strip().upper(), **order}


# Only the functions listed here can ever be run. The AI cannot run anything else.
AVAILABLE_TOOLS = {"check_order_status": check_order_status}

# ---------- 3. Describe the tool to the AI ----------
order_tool = types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="check_order_status",
        description="Get the status of a customer order using its order ID, for example ORD-101.",
        parameters={
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "The order ID, for example ORD-101"},
            },
            "required": ["order_id"],
        },
    )
])

config = types.GenerateContentConfig(
    system_instruction=(
        "You are a customer support assistant for an online store. "
        "Use the check_order_status tool for questions about a specific order. "
        "If the tool says an order was not found, tell the customer and ask them to check the ID. "
        "Never invent order details."
    ),
    tools=[order_tool],
)


# ---------- 4. The tool-calling flow ----------
def ask(user_message):
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=user_message)])]

    # Step A: send the request. The AI decides if a tool is needed.
    response = client.models.generate_content(model=MODEL, contents=contents, config=config)

    if not response.function_calls:
        print("\n[No tool needed] AI answers directly:")
        print(response.text)
        return

    # Step B: the AI asks for a tool. Our application runs it.
    result_parts = []
    for call in response.function_calls:
        print(f"\n[1] AI requests tool: {call.name}({dict(call.args)})")

        function = AVAILABLE_TOOLS.get(call.name)
        if function is None:
            result = {"error": f"Unknown tool: {call.name}"}
        else:
            result = function(**call.args)
        print(f"[2] Application ran the function and got: {result}")

        result_parts.append(types.Part.from_function_response(name=call.name, response=result))

    # Step C: send the result back so the AI can write the final answer.
    contents.append(response.candidates[0].content)           # the AI's tool request
    contents.append(types.Content(role="user", parts=result_parts))  # our function result
    final = client.models.generate_content(model=MODEL, contents=contents, config=config)
    print(f"[3] Final AI response: {final.text}")


print("Order Support Assistant (type 'quit' to exit)")
print("Try order IDs: ORD-101, ORD-102, ORD-103")
while True:
    message = input("\nYou: ").strip()
    if message.lower() in ("quit", "exit"):
        break
    if not message:
        continue
    try:
        ask(message)
    except Exception as e:
        print("Request failed:", e)
