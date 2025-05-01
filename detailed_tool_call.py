from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()
llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b."""
    return a * b

@tool   
def add(a: int, b: int) -> int:
    """Add a and b."""
    return a + b

tools = [multiply, add]
model_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools)

def should_continue(state: MessagesState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

def call_model(state: MessagesState):
    messages = state["messages"]    
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}

workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, ["tools", END])
workflow.add_edge("tools", "agent")
app = workflow.compile()

ans = app.invoke({"messages": [HumanMessage(content="multiply 20 and 30 and add answer to 10")]})
final_response = ans["messages"][-1]
print(final_response.content)