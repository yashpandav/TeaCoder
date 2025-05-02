from cursor_agent_conf import app, SYSTEM_PROMPT
from langchain_core.messages import SystemMessage, HumanMessage

while True:
    user_input = input("💬 What do you want the assistant to do? ")
    if user_input.strip().lower() in {"exit", "quit"}:
        print("👋 Goodbye!")
        break
    
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_input)
    ]
    
    # Track error attempts to prevent infinite loops
    error_attempts = 0
    max_error_attempts = 3

    while True:
        ans = app.invoke({"messages": messages}, config={
            "recursion_limit": 50
        })
        final_response = ans["messages"][-1]
        print("🤖", final_response.content)
        
        if isinstance(final_response.content, str) and "The previous command resulted in an error" in final_response.content:
            error_attempts += 1
            if error_attempts >= max_error_attempts:
                print("Maximum error correction attempts reached. Please provide new instructions.")
                break
        else:
            break
