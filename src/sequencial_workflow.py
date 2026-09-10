# Sequential LangGraph workflow that turns messy raw text into a polished final script.
# Stage 1 (editor) cleans grammar and spelling; stage 2 (script) turns that into a script.
# Stage 3 (final_output) produces the last version from the script, then the graph ends.
# Nodes run in a fixed chain: START → editor → script → final_output → END.

import os
from typing import TypedDict, List
from rich import print

class pipelinestate(TypedDict):
    raw_input: str
    edited_text: str  
    script_text: str
    final_output: str

from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b", api_key=os.getenv("GROQ_API_KEY"), temperature=0.7)

# response = llm.invoke("what is the capital of California?")
# print(response.content)

def editor_node(state: pipelinestate) -> dict:
    # Use the LLM to edit the raw input text
    """Stage 1: Clean and edit the raw input text and grammar."""
    prompt = f"Please clean and edit the following text for grammar and clarity. Fix the grammtical error and spelling mistake and make a polish clean text Return only edited text:\n\n{state['raw_input']}"
    llm_response = llm.invoke(prompt)
    return {"edited_text": llm_response.content.strip()}

def script_node(state: pipelinestate) -> dict:
    # Use the LLM to generate a script based on the edited text
    """Stage 2: Generate a script based on the edited text."""
    prompt = f"Please generate a script based on the following edited text:\n\n{state['edited_text']}"
    llm_response = llm.invoke(prompt)
    return {"script_text": llm_response.content.strip()}

def final_output_node(state: pipelinestate) -> dict:
    # Use the LLM to generate the final output based on the script
    """Stage 3: Generate the final output based on the script in Hinglish."""
    prompt = f"Please generate the final output based on the following script:\n\n{state['script_text']}"
    llm_response = llm.invoke(prompt)
    return {"final_output": llm_response.content.strip()}

from langgraph.graph import StateGraph, START, END
graph = StateGraph(pipelinestate)

graph.add_node("editor", editor_node)
graph.add_node("script", script_node)
graph.add_node("final_output", final_output_node)

graph.add_edge(START, "editor")
graph.add_edge("editor", "script")
graph.add_edge("script", "final_output")
graph.add_edge("final_output", END)

app = graph.compile()
result = app.invoke({"raw_input": "AI agents are the future of tech. They can think, plan and act on their own. LangGraph help you build these agents with proper control and memory."})

print("Your Result:")
print("Final Output:", result["final_output"])