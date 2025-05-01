from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from dataclasses import dataclass
import subprocess
import os
load_dotenv()

class ToolSelector:
    use_command_exec = False
    use_read_file = False
    use_write_file = False
    use_scan_directory = False
    use_analyze_code = False

SYSTEM_PROMPT = """
You are TeaCoder, an expert AI coding assistant specialized in full-stack development. 

You have deep expertise in:
- Frontend: React, Vue, Angular, HTML/CSS, JavaScript/TypeScript
- Backend: Node.js, Python (Django, Flask), Java (Spring), Ruby on Rails
- Database: SQL, MongoDB, Firebase
- DevOps: Docker, CI/CD, AWS, deployment processes

Tools You Can Use:
- command_exec: Executes shell commands (string input only, not JSON)
- read_file: Reads a file at a specified path
- write_file: Writes content to a specified file path
- scan_directory: Lists files in a given directory
- analyze_code: Analyzes source code for structure and logic

INSTRUCTIONS:
1. Maintain a model of the project structure and files.
2. Suggest appropriate architecture and best practices.
3. Generate complete, working code when needed.
4. Execute terminal commands to install dependencies, create files, or run builds.
5. Modify existing code in context when adding features.
6. Provide clear explanations for your decisions.

Rules:
For tasks that require multiple steps (like creating a complete project), make sure you execute ALL necessary actions one after another
- For example, when creating an Express.js app:
    1. Create directory and initialize npm
    2. Install dependencies
    3. Create server.js with complete code
    4. Create other necessary files (routes, controllers, etc.)   
- IMPORTANT: NEVER attempt to run a project. Only SUGGEST how to run the project, for example:
    - "To run the project, you can use: npm run dev"
    - "You can start the server with: python manage.py runserver"
    - "Launch the application with: java -jar myapp.jar"
    - "Start the app with: ruby myapp.rb"
- DO NOT use command_exec to run any of these commands; ONLY suggest them to the user
- DO NOT stop after just one action - ANALYZE the result and CONTINUE until the task is COMPLETE
- Perform one step at a time and wait for next input
- Analyze existing code before modifying it
- Ensure commands are appropriate for the current OS (Windows assumed)
- When asked to build something, create proper file structures and all necessary files
-When using write_file:
    - "path": Full file path (e.g., "src/math.js")
    - "content": File contents as string
- When reading files:
    - Always specify full paths
    - Handle encoding automatically
    - Display contents in readable format
- Command to be executed must be a string, if it is dictionary than find the command string from the dictionary.
- npx-create-raect-app is not supported, use npm create vite@latest instead.
- When on Windows:
    - Use 'rmdir /s /q directory_name' instead of 'rm -rf directory_name' for deleting directories
    - Use 'del filename' instead of 'rm filename' for deleting files
    - Use 'type filename' instead of 'cat filename' for displaying file contents

- For React app creation:
    - Always use interactive commands like: npm create vite@latest my-app
    - use interactive commands that prompt for user input
    - After creating the app, install dependencies but DO NOT run the app
    - ONLY suggest the command to run the app (e.g., "You can start the app with: npm run dev")

Error Handling Format:
{
    "step": "output",
    "content": "Error: Unable to read file. Ensure path is correct and file exists."
}

DO NOT:
- Output raw logs or terminal noise
- Use tools without `step: action`
- Skip next steps if more work is needed
- Attempt to run or start projects - only suggest the commands

"""

llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

@tool
def command_exec(command: str) -> str:
    """Execute a shell command and return its output."""
    print("🔑 ", command)
    
    try:
        result = subprocess.run(command, shell=True, check=True, text=True)
        return result.stdout or "Command executed successfully with no output."
    except subprocess.CalledProcessError as e:
        return f"Error:\n{e.stderr or str(e)}"

@tool   
def read_file(file_path: str) -> str:
    """Read a file."""
    print("🔑 ", file_path)
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            print(f"📄 Read file: {file_path}")
            print("📜 Content:")
            print("-" * 40)
            print(content)
            print("-" * 40)
            return content
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='latin-1') as file:
                content = file.read()
                print(f"📄 Read file (latin-1 encoding): {file_path}")
                print("📜 Content:")
                print("-" * 40)
                print(content)
                print("-" * 40)
                return content
        except Exception as e:
            return f"Error reading file: {e}"

@tool
def write_file(file_path: str, content: str) -> str:
    """Write to a file."""
    print("🔑 ", file_path)
    print("🔑 ", content)
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
            print(f"📝 Wrote to file: {file_path}")
            return f"File {file_path} written successfully"
    except Exception as e:
        return f"Error writing file: {e}"
    
@tool
def scan_directory(directory: str) -> str:
    """Scan a directory."""
    print("🔑 ", directory)
    try:
        files = os.listdir(directory)
        print(f"📂 Scanned directory: {directory} with files: {files}")
        return files
    except Exception as e:
        return f"Error scanning directory: {str(e)}"
    
@tool
def analyze_code(file_path: str) -> str:
    """Analyze a file."""
    print("🔑 ", file_path)
    with open(file_path, 'r') as file:  
        content = file.read()
        return f"Analyzed code in {file_path}."

tools = [command_exec, read_file, write_file, scan_directory, analyze_code]
model_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools)

def should_continue(state: MessagesState):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

def call_model(state: MessagesState):
    messages = state["messages"]    
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}

def handle_tool_result(state: MessagesState):
    last_tool_result = state["messages"][-1]

    if isinstance(last_tool_result.content, str) and "Error" in last_tool_result.content:
        # Instead of just showing the error, add context so the model can try to fix it
        return {
            "messages": [
                AIMessage(content=f"The previous command resulted in an error:\n{last_tool_result.content}\nPlease analyze this error and try a different approach to solve the same task.")
            ]
        }
    
    return {"messages": [last_tool_result]}


tool_selector = ToolSelector()

workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_node("handle_tool", handle_tool_result)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, ["tools", END])
workflow.add_edge("tools", "handle_tool")
workflow.add_edge("handle_tool", "agent")

app = workflow.compile()

while True:
    user_input = input("💬 What do you want the assistant to do? ")
    if user_input.strip().lower() in {"exit", "quit"}:
        print("👋 Goodbye!")
        break
    
    # Initialize the messages list with system prompt and user input
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_input)
    ]
    
    # Track error attempts to prevent infinite loops
    error_attempts = 0
    max_error_attempts = 3
    
    # Run the workflow until completion or max error attempts
    while True:
        ans = app.invoke({"messages": messages})
        final_response = ans["messages"][-1]
        print("🤖", final_response.content)
        
        # Check if the response indicates an error
        if isinstance(final_response.content, str) and "The previous command resulted in an error" in final_response.content:
            error_attempts += 1
            if error_attempts >= max_error_attempts:
                print("⚠️ Maximum error correction attempts reached. Please provide new instructions.")
                break
        else:
            # No error, exit the retry loop
            break
