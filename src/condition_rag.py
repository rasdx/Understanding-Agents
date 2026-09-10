# College assistant that classifies a student query as academic, fee, or general.
# Academic and fee questions retrieve context from the handbook PDFs via FAISS RAG.
# A LangGraph routes the query to the matching retriever, then the LLM answers from that context.
# General chat skips retrieval and replies conversationally in a simple CLI loop.


import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

llm = ChatGroq(model="openai/gpt-oss-120b",api_key=os.getenv("GROQ_API_KEY"),temperature=0.7,)

def build_retriever(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    docs = splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 4})


academic_retriever = build_retriever("/Users/ayushirastogi/Documents/AgenticAI/pdfs/academics_handbook.pdf")
fee_retriever = build_retriever("/Users/ayushirastogi/Documents/AgenticAI/pdfs/fee_structure.pdf")


class RagState(TypedDict):
    programme: str
    messages: Annotated[list, add_messages]
    query_type: str
    retriever_context: str


def classifier_node(state: RagState):

    last_message = state["messages"][-1].content

    prompt = (
        "Classify the following student query into exactly one category: "
        "'academic', 'fee', or 'general'.\n\n"

        "Use 'academic' for questions about attendance, exams, grading, "
        "credits, promotion, course structure, summer training, or degree requirements.\n"

        "Use 'fee' for questions about tuition, payment, refund, late charges, "
        "scholarships, or any money-related topic.\n"

        "Use 'general' for greetings, casual talk, or anything unrelated to "
        "college rules or fees.\n\n"

        f"Query: {last_message}\n\n"

        "Return only one word: academic, fee, or general."
    )

    response = llm.invoke(prompt)

    category = response.content.strip().lower()

    if "academic" in category:
        category = "academic"
    elif "fee" in category:
        category = "fee"
    else:
        category = "general"

    return {
        "query_type": category
    }

def academic_rag_node(state: RagState):

    last_message = state["messages"][-1].content
    docs = academic_retriever.invoke(last_message)
    context = "\n\n".join(doc.page_content for doc in docs)
    return {"retriever_context": context}


def fee_rag_node(state: RagState):

    last_message = state["messages"][-1].content
    docs = fee_retriever.invoke(last_message)
    context = "\n\n".join(doc.page_content for doc in docs)
    return {
        "retriever_context": context
    }


def general_node(state: RagState):
    return {"retriever_context": ""}


def response_node(state: RagState):

    question = state["messages"][-1].content
    query_type = state["query_type"]
    context = state["retriever_context"]
    if query_type == "academic":

        prompt = f"""You are a helpful university assistant. Answer ONLY from the given context.
        Context:
        {context}
        Question:
        {question}
        If the answer is not present in the context, say: "I couldn't find this information in the academic handbook."""

    elif query_type == "fee":

        prompt = f"""You are a helpful university assistant. Answer ONLY from the given context.
        Context:
        {context}
        Question:
        {question}
        If the answer is not present in the context, say:"I couldn't find this information in the fee handbook."""

    else:
        prompt = f"""You are a friendly university assistant.Answer the student's question naturally.

Question:
{question}
"""

    response = llm.invoke(prompt)

    return {
        "messages": [
            {
                "role": "assistant",
                "content": response.content
            }
        ]
    }


def route_query(state: RagState):
    return state["query_type"]

builder = StateGraph(RagState)

builder.add_node("classifier", classifier_node)
builder.add_node("academic", academic_rag_node)
builder.add_node("fee", fee_rag_node)
builder.add_node("general", general_node)
builder.add_node("response", response_node)

builder.add_edge(START, "classifier")

builder.add_conditional_edges(
    "classifier",
    route_query,
    {
        "academic": "academic",
        "fee": "fee",
        "general": "general",
    },
)

builder.add_edge("academic", "response")
builder.add_edge("fee", "response")
builder.add_edge("general", "response")

builder.add_edge("response", END)

app = builder.compile()

print("=" * 60)
print("🎓 College Assistant")
print("Courses offered: BBA, BCA, B.Com (H)")
print("Type 'exit' to quit.")
print("=" * 60)

while True:

    query = input("\nYou: ")

    if query.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    initial_state = {
        "programme": "BCA",
        "messages": [HumanMessage(content=query)],
        "query_type": "",
        "retriever_context": "",
    }

    result = app.invoke(initial_state)

    print("\nAssistant:")
    print(result["messages"][-1].content)
