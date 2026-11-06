EXIT_COMMANDS = {"bye", "goodbye", "exit", "quit"}
BOT_NAME = "NOVA"
def chatbot_response(user_input: str) -> str:

    message = user_input.strip().lower()

    if message in {"hello", "hi", "hey"}:
        return "Hello! How can I help you?"
    if "your name" in message or "who are you" in message:
        return (
            f"I am {BOT_NAME}, "
            "a simple AI chatbot created using Python."
        )
    if "how are you" in message:
        return "I'm doing great! Thanks for asking."

    if "help" in message:
        return (
            "Sure! You can ask me about my name, "
            "capabilities, Python, or say hello."
        )

    if "what can you do" in message:
        return (
            "I can respond to basic questions "
            "using predefined conversational rules."
        )

    if "python" in message:
        return (
            "Python is a powerful programming language "
            "widely used in AI and data science."
        )

    if "thank" in message:
        return "You're welcome!"

    if message in EXIT_COMMANDS:
        return "Goodbye! Have a great day!"

    return (
        "I'm sorry, I don't understand that yet. "
        "Try asking me something else."
    )
def main() -> None:
    print("=" * 50)
    print("              NOVA - AI CHATBOT")
    print("=" * 50)
    print("Type 'bye', 'exit', or 'quit' to end the conversation.")
    print()

    while True:
        user_input = input("You: ")

        response = chatbot_response(user_input)

        print(f"{BOT_NAME}: {response}")

        if user_input.strip().lower() in EXIT_COMMANDS:
            break
if __name__ == "__main__":
    main()