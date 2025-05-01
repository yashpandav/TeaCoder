import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from langsmith.wrappers import wrap_openai
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from typing import Literal
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model

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
tool_node = ToolNode(tools)

llm_with_tools = llm.bind_tools(tools)

ans = tool_node.invoke({"messages": [llm_with_tools.invoke("Multiply 20 and 30")]})

print(ans["messages"][0].content)