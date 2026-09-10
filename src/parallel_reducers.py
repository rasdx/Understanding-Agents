# Safety analyzer that scores a script for toxicity, copyright risk, and cultural sensitivity.
# Three LangGraph nodes run in parallel from START, each returning a 0–100 score.
# A reducer merges those score dicts into one safety_score on shared state.
# The compiled graph is invoked on a sample script and prints the combined scores.

import os
from typing import TypedDict, Annotated

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from rich import print

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.1,
)

class Score(BaseModel):
    score: int = Field(..., ge=0, le=100)

structured_llm = llm.with_structured_output(Score)

def merge_score(existing: dict[str, int], new: dict[str, int]) -> dict[str, int]:
    if existing is None:
        return new
    return {**existing, **new}

class AnalyzerState(TypedDict):
    raw_text: str
    safety_score: Annotated[dict[str, int], merge_score]

def toxicity_analyzer(state: AnalyzerState):
    prompt = f"""Analyze the following text for toxicity. Return a score between 0 and 100.

        Text:
        {state["raw_text"]}"""
    
    result = structured_llm.invoke(prompt)
    return {"safety_score": {"toxicity_level": result.score}}

def copyright_analyzer(state: AnalyzerState):
    prompt = f"""Analyze the following text for copyright risk. Return a score between 0 and 100.
Text:
{state["raw_text"]}
"""
    
    result = structured_llm.invoke(prompt)
    return {"safety_score": {"copyright_risk": result.score}}

def culture_analyzer(state: AnalyzerState):
    prompt = (
    "Analyze the following text for regional sensitivities, political landmines, "
    "or cultural insensitivity that might offend a global audience.\n\n"
    "Provide a score from 0 to 100, where:\n"
    "- 0 means completely safe\n"
    "- 100 means highly offensive\n\n"
    "Return ONLY the plain integer number. Do not return any explanation, JSON, or extra text.\n\n"
    f"Text:\n{state['raw_text']}"
)
    result = structured_llm.invoke(prompt)
    return {"safety_score": {"cultural_sensitivity": result.score}}

builder = StateGraph(AnalyzerState)

builder.add_node("toxicity", toxicity_analyzer)
builder.add_node("copyright", copyright_analyzer)
builder.add_node("culture", culture_analyzer)

builder.add_edge(START, "toxicity")
builder.add_edge(START, "copyright")
builder.add_edge(START, "culture")

builder.add_edge("toxicity", END)
builder.add_edge("copyright", END)
builder.add_edge("culture", END)

app = builder.compile()

sample_script = """
Yo guys! Welcome back to the stream. Today I am going to show you how to hack into
your friend's system using a script I copied directly from an online forum.
Honestly, traditional security protocols are absolute garbage and anyone still using
them is an absolute idiot. Let's dive into the code!
"""

initial_state = {
    "raw_text": sample_script,
    "safety_score": {},
}

final_out = app.invoke(initial_state)

print(final_out["safety_score"])