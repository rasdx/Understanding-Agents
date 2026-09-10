- HumanIntheLoop

# LinkedIn post generator that pauses for a human to approve or reject each draft.

# A writer LLM drafts the post, then LangGraph interrupt waits for your review.

# If you give feedback, the writer rewrites; if you approve, the workflow ends.

# MemorySaver checkpoints the run so the loop can resume after each human reply.

- Iterative-tools

# LinkedIn post generator that writes, reviews, and rewrites until the post is approved.

# The writer can call Tavily search for current facts before producing a draft.

# A stricter reviewer LLM scores the draft and either approves it or sends feedback back.

# LangGraph loops writer → tools/extract → reviewer up to three attempts, then prints the final post.

- parallel-reducers

# Safety analyzer that scores a script for toxicity, copyright risk, and cultural sensitivity.

# Three LangGraph nodes run in parallel from START, each returning a 0–100 score.

# A reducer merges those score dicts into one safety_score on shared state.

# The compiled graph is invoked on a sample script and prints the combined scores.

- seq_wrkflow

# Sequential LangGraph workflow that turns messy raw text into a polished final script.

# Stage 1 (editor) cleans grammar and spelling; stage 2 (script) turns that into a script.

# Stage 3 (final_output) produces the last version from the script, then the graph ends.

# Nodes run in a fixed chain: START → editor → script → final_output → END.

- states

# Reference examples of how to define LangGraph-style shared state in Python.

# TypedDict: simple typed dictionary with no runtime validation.

# Pydantic BaseModel: same fields with descriptions and checks when data is created.

# Dataclass: a lightweight class holding name, abbreviation, capital, population, and area.
# Understanding-Agents
# Understanding-Agents
# Understanding-Agents
