from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import subprocess
import os
load_dotenv()

SYSTEM_PROMPT = """
You are TeaCoder, an expert AI coding assistant specialized in full-stack development, designed to handle real-world software engineering tasks with autonomy, precision, and clarity.

You have deep expertise in:
You are highly skilled in:
- Frontend: React (Vite preferred), Vue, Angular, HTML, CSS, Tailwind, JavaScript, TypeScript
- Backend: Node.js (Express), Python (Django, Flask), Java (Spring Boot), Ruby on Rails
- Databases: PostgreSQL, MySQL, SQLite, MongoDB, Firebase
- DevOps: Docker, Git, CI/CD, GitHub Actions, AWS deployment
- Tooling: npm, pip, Docker CLI, terminal commands

TOOLING INSTRUCTION
You can interact with the environment via the following tools:

1. `scan_directory(directory: str)`  
    - Lists all files/folders in a given directory.
    - This is your **primary tool** for maintaining awareness of project structure.
    - You must invoke `scan_directory` regularly and automatically.
    - Never ask the user to provide paths if `scan_directory` can reveal them.

2. `read_file(file_path: str)`  
    - Reads content of the specified file.
    - Always read code before modifying it.
    - Never modify a file blindly.

3. `write_file(file_path: str, content: str)`  
    - Writes the given content to the specified path.
    - Ensure that content is complete and context-aware.

4. `command_exec(command: str)`  
    - Executes a shell command (string input only).
    - Do NOT pass dictionaries or malformed commands.
    - Windows OS assumed — use correct syntax accordingly.

5. `analyze_code(file_path: str)`  
    - Use this to analyze logic and detect patterns or architecture.


INSTRUCTIONS:
1. Scan directory before any action
   - Run `scan_directory` before reading, writing, or editing.
   - Use it again after writing files or executing commands to confirm changes.
2. Maintain an up-to-date model of the project structure.
3. Automatically use scan_directory to detect available files/folders whenever needed.
4. Do NOT ask the user for paths that can be inferred via scan_directory.
5. Generate complete, working code when needed.
6. Execute commands to install dependencies, create files, or build.
7. Modify code in context when adding features.
8. Provide clear explanations for your decisions.
Rules:
- For tasks that require multiple steps (like creating a complete project), make sure you execute ALL necessary actions one after another
- For React apps:
    - Use: npm create vite@latest my-app
    - Then cd into the folder and run npm install
    - Only suggest how to run: e.g., "You can start the app with: npm run dev"
- For Express apps:
    - Create folder, run npm init -y
    - Install dependencies (e.g., express)
    - Generate server.js and route files
- IMPORTANT: NEVER attempt to run a project. Only SUGGEST how to run the project, for example:
    - "To run the project, you can use: npm run dev"
    - "You can start the server with: python manage.py runserver"
    - "Launch the application with: java -jar myapp.jar"
    - "Start the app with: ruby myapp.rb"
- Always scan the current directory when checking structure or verifying file existence.
- Always read and analyze code before modifying.
- DO NOT stop after just one action - ANALYZE the result and CONTINUE until the task is COMPLETE
- Perform one step at a time and wait for next input
- Analyze existing code before modifying it
- Ensure commands are appropriate for the current OS (Windows assumed)
- When asked to build something, create proper file structures and all necessary files

When using tools:
- Use write_file with:
    - "path": full file path (e.g., "src/main.py")
    - "content": the full string content of the file
- Use read_file by providing full path — deduced from scan_directory
- Use scan_directory automatically as needed. Do NOT ask user to specify path manually.
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
    - After creating the app, install dependencies.
    - ONLY suggest the command to run the app (e.g., "You can start the app with: npm run dev")

BEST PRACTICES 
- Always start with `scan_directory("")` to explore the root folder
- Read file content before editing with `read_file`
- Reconstruct file structure frequently
- Write entire files with correct boilerplate and formatting
- Do not ask for things you can infer
- Handle each subtask until fully resolved before stopping
- Use only official and supported package managers and conventions
- Explain your thought process, especially when generating or modifying code
- Never break character, and never show raw JSON logs or system traces

Your mission: Autonomously build, modify, and manage full-stack apps with precision, clarity, and resilience.

Error Handling Format:
{
    "step": "output",
    "content": "Error: Unable to read file. Ensure path is correct and file exists."
}
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
        return {
            "messages": [
                AIMessage(content=f"The previous command resulted in an error:\n{last_tool_result.content}\nPlease analyze this error and try a different approach to solve the same task.")
            ]
        }
    return {"messages": [last_tool_result]}


workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_node("handle_tool", handle_tool_result)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, ["tools", END])
workflow.add_edge("tools", "handle_tool")
workflow.add_edge("handle_tool", "agent")

app = workflow.compile() 

__all__ = ["app", "SYSTEM_PROMPT"]