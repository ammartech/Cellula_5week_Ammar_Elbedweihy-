# app.py
"""
Unified NLP Internship Project App
----------------------------------

This single-file Streamlit application combines:

- Week 3 : General LLM Chat
- Week 4 : System Prompting, Memory, Chroma RAG, RAG Evaluation, DeepEval hook
- Week 5 : LangGraph-powered Python Code Assistant with RAG over code examples

You can run this file using:

    streamlit run app.py

Make sure you install the required dependencies first:

    pip install streamlit langchain langchain-community langgraph chromadb sentence-transformers

Optional (for DeepEval and OpenAI-compatible LLMs):

    pip install deepeval openai

Notes:
- By default this file uses `ChatOpenAI` from LangChain (which expects an OPENAI_API_KEY or
  compatible API). You can replace it with a local LLM wrapper (e.g., llama-cpp-python)
  inside the `get_llm()` function.
"""

import os
from typing import Dict, Any, List, TypedDict, Literal, Optional, Tuple

import streamlit as st

# LangChain / LangGraph imports
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationalRetrievalChain

from langgraph.graph import StateGraph, END

# ---------------------------------------------------------------------------
# 0. GLOBAL CONFIG & HELPERS
# ---------------------------------------------------------------------------

# Paths for persisted vector stores
DOCS_CHROMA_DIR = "./chroma_docs_db"
CODE_CHROMA_DIR = "./chroma_code_db"

# Embeddings model
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_llm(system_prompt: str = ""):
    """
    Return a ChatOpenAI-based LLM (or a local LLM if you replace this function).

    - If OPENAI_API_KEY is set, ChatOpenAI will use it.
    - Otherwise, we still create ChatOpenAI object. If the call fails, UI will show an error.

    You can replace this with a local LLM, e.g.:

        from langchain_community.llms import LlamaCpp
        return LlamaCpp(
            model_path="your_model.gguf",
            temperature=0.1,
            n_ctx=4096,
        )

    but keep the interface compatible (invoke(prompt) -> message).
    """
    # Note: system_prompt is injected manually in the prompts we build later.
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",  # Change to your available model
        temperature=0.2,
    )
    return llm


@st.cache_resource(show_spinner=False)
def get_embeddings():
    """
    Cached HuggingFace sentence-transformer embeddings.
    """
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)


def ensure_dir(path: str):
    """
    Create directory if it does not exist.
    """
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. SAMPLE DOCUMENTS FOR RAG (WEEK 4)
# ---------------------------------------------------------------------------

def build_sample_docs() -> List[Document]:
    """
    Build a simple set of demonstration documents to be stored in Chroma.
    In a real project, you would load PDFs, text files, etc.
    """
    raw_docs = [
        {
            "title": "RAG Overview",
            "content": (
                "Retrieval-Augmented Generation (RAG) is a technique that combines "
                "a retrieval component with a generative model. It allows the model "
                "to ground its answers in external knowledge sources such as documents "
                "or databases."
            ),
        },
        {
            "title": "LangChain Memory",
            "content": (
                "LangChain provides different types of memory, including buffer "
                "memory, summary memory, and entity memory. Memory helps the LLM "
                "maintain context across multiple turns in a conversation."
            ),
        },
        {
            "title": "Chroma Vector Store",
            "content": (
                "Chroma is a simple and powerful open-source embedding database. "
                "It can be used as a persistent vector store for RAG applications "
                "and is well integrated with LangChain."
            ),
        },
        {
            "title": "LangGraph State Machines",
            "content": (
                "LangGraph is a library that helps define state machines for LLM "
                "applications. It uses nodes and edges to handle multi-step logic "
                "in a structured way."
            ),
        },
        {
            "title": "RAG Evaluation Metrics",
            "content": (
                "Retrieval quality in RAG systems can be measured using metrics such "
                "as Precision@K, Recall@K, Mean Reciprocal Rank (MRR), and normalized "
                "Discounted Cumulative Gain (nDCG)."
            ),
        },
    ]
    docs: List[Document] = []
    for d in raw_docs:
        docs.append(
            Document(
                page_content=d["content"],
                metadata={"title": d["title"]},
            )
        )
    return docs


@st.cache_resource(show_spinner=False)
def build_docs_chroma() -> Chroma:
    """
    Create or load a Chroma vector store for the sample documents.
    """
    ensure_dir(DOCS_CHROMA_DIR)
    embeddings = get_embeddings()
    docs = build_sample_docs()
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=DOCS_CHROMA_DIR,
    )
    vectordb.persist()
    return vectordb


# ---------------------------------------------------------------------------
# 2. RAG CHAIN (ConversationalRetrievalChain)
# ---------------------------------------------------------------------------

def build_rag_chain(system_prompt: str = "") -> ConversationalRetrievalChain:
    """
    Build a conversational RAG chain using LangChain and Chroma.

    This corresponds to Week 4 tasks:
    - System Prompting
    - Memory
    - Chroma Vector DB
    """
    vectordb = build_docs_chroma()
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    if not system_prompt:
        system_prompt = (
            "You are a helpful AI assistant specialized in RAG, LangChain, and LangGraph. "
            "Use the retrieved context to answer questions accurately. "
            "If you are unsure, say that you are not sure."
        )

    llm = get_llm()
    memory = ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=1000,
        memory_key="chat_history",
        return_messages=True,
    )

    # We will build the chain manually through ConversationalRetrievalChain.from_llm
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=False,
    )
    # We'll inject system instructions at query time inside `rag_answer()`.
    chain._system_prompt = system_prompt  # custom attribute for reference
    return chain


@st.cache_resource(show_spinner=False)
def get_rag_chain() -> ConversationalRetrievalChain:
    return build_rag_chain()


def rag_answer(question: str, chat_history: Optional[List[Tuple[str, str]]] = None) -> Dict[str, Any]:
    """
    Use the RAG chain to answer a question.

    Returns a dict with:
      - answer: str
      - docs: List[str] (retrieved contexts)
    """
    chain = get_rag_chain()
    # ConversationalRetrievalChain expects 'question' and 'chat_history'
    if chat_history is None:
        chat_history = []
    result = chain(
        {"question": question, "chat_history": chat_history}
    )
    # We also extract retrieved docs from the retriever if needed:
    vectordb = build_docs_chroma()
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    retrieved_docs = retriever.get_relevant_documents(question)

    return {
        "answer": result["answer"],
        "docs": [d.page_content for d in retrieved_docs],
    }


# ---------------------------------------------------------------------------
# 3. RAG EVALUATION METRICS (Precision@K, Recall@K, MRR, nDCG)
# ---------------------------------------------------------------------------

def precision_at_k(retrieved: List[int], relevant: List[int], k: int) -> float:
    """
    Compute Precision@K given:
      - retrieved: list of doc IDs in retrieved order
      - relevant: list of doc IDs that are relevant
    """
    retrieved_k = retrieved[:k]
    if not retrieved_k:
        return 0.0
    hits = sum(1 for doc_id in retrieved_k if doc_id in relevant)
    return hits / len(retrieved_k)


def recall_at_k(retrieved: List[int], relevant: List[int], k: int) -> float:
    """
    Compute Recall@K.
    """
    if not relevant:
        return 0.0
    retrieved_k = retrieved[:k]
    hits = sum(1 for doc_id in retrieved_k if doc_id in relevant)
    return hits / len(relevant)


def mrr(retrieved: List[int], relevant: List[int]) -> float:
    """
    Mean Reciprocal Rank for a single query (we use it on one query here).
    """
    for idx, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / idx
    return 0.0


def dcg(scores: List[float]) -> float:
    """
    Discounted Cumulative Gain.
    'scores' are relevance scores ordered by rank (highest rank first).
    """
    import math
    total = 0.0
    for i, s in enumerate(scores, start=1):
        total += (2**s - 1) / math.log2(i + 1)
    return total


def ndcg_at_k(retrieved: List[int], relevant: Dict[int, float], k: int) -> float:
    """
    Compute nDCG@K.

    - retrieved: ordered list of doc IDs
    - relevant: mapping doc_id -> relevance score (e.g., 0, 1, 2)
    """
    retrieved_k = retrieved[:k]
    scores = [relevant.get(doc_id, 0.0) for doc_id in retrieved_k]
    ideal_scores = sorted(relevant.values(), reverse=True)[:k]

    actual_dcg = dcg(scores)
    ideal_dcg = dcg(ideal_scores) if ideal_scores else 1.0
    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def demo_rag_evaluation() -> Dict[str, float]:
    """
    A tiny demo of retrieval evaluation with dummy data.

    Suppose:
      retrieved docs IDs = [2, 0, 3, 1]
      relevant docs IDs = [0, 3]
      graded relevance: {0: 2, 3: 1}

    Returns a dict of metric values.
    """
    retrieved = [2, 0, 3, 1]
    relevant_ids = [0, 3]
    graded = {0: 2.0, 3: 1.0}

    k = 3
    prec = precision_at_k(retrieved, relevant_ids, k)
    rec = recall_at_k(retrieved, relevant_ids, k)
    r_mrr = mrr(retrieved, relevant_ids)
    r_ndcg = ndcg_at_k(retrieved, graded, k)

    return {
        "Precision@3": prec,
        "Recall@3": rec,
        "MRR": r_mrr,
        "nDCG@3": r_ndcg,
    }


# ---------------------------------------------------------------------------
# 4. DEEPEVAL HOOK (OPTIONAL)
# ---------------------------------------------------------------------------

def try_deepeval_example() -> Optional[Dict[str, Any]]:
    """
    Try to run a simple DeepEval example.
    If deepeval is not installed, return None.
    """
    try:
        from deepeval.metrics import AnswerRelevancyMetric
        from deepeval.test_case import LLMTestCase
    except ImportError:
        return None

    # Simple test case
    test_case = LLMTestCase(
        input="What is RAG?",
        actual_output="RAG combines retrieval and generation for better grounded answers.",
        expected_output="RAG stands for Retrieval-Augmented Generation and uses external documents.",
        context=[
            "Retrieval-Augmented Generation (RAG) combines a retriever with a generator model."
        ],
    )
    metric = AnswerRelevancyMetric()
    score = metric.measure(test_case)
    return {
        "metric_name": "AnswerRelevancy",
        "score": score,
    }


# ---------------------------------------------------------------------------
# 5. WEEK 5 – LANGGRAPH CODE ASSISTANT
# ---------------------------------------------------------------------------

class CodeAssistantState(TypedDict, total=False):
    user_input: str
    chat_state: Literal["generate", "explain"]
    retrieved_examples: List[str]
    llm_response: str


def build_code_examples() -> List[Document]:
    """
    Build a small in-memory corpus of Python code examples.
    These will be used in the code-assistant RAG (LangGraph side).
    """
    samples = [
        {
            "title": "sum_even_numbers",
            "description": "Python function that sums even numbers in a list.",
            "code": """def sum_even_numbers(nums):
    return sum(n for n in nums if n % 2 == 0)
""",
        },
        {
            "title": "factorial_recursive",
            "description": "Recursive factorial implementation in Python.",
            "code": """def factorial(n: int) -> int:
    if n <= 1:
        return 1
    return n * factorial(n - 1)
""",
        },
        {
            "title": "file_reader",
            "description": "Safely read a UTF-8 text file and return its contents.",
            "code": """from pathlib import Path

def read_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f\"File not found: {path}\")
    return p.read_text(encoding=\"utf-8\")
""",
        },
        {
            "title": "simple_logger",
            "description": "Basic logging setup for a small Python app.",
            "code": """import logging

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format=\"[%(asctime)s] %(levelname)s - %(message)s\",
    )
    return logging.getLogger(__name__)
""",
        },
    ]
    docs: List[Document] = []
    for s in samples:
        docs.append(
            Document(
                page_content=s["description"] + "\n\n" + s["code"],
                metadata={"title": s["title"]},
            )
        )
    return docs


@st.cache_resource(show_spinner=False)
def build_code_chroma() -> Chroma:
    """
    Create or load a Chroma vector store for code examples.
    This is specifically for the LangGraph code assistant.
    """
    ensure_dir(CODE_CHROMA_DIR)
    embeddings = get_embeddings()
    docs = build_code_examples()
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CODE_CHROMA_DIR,
    )
    vectordb.persist()
    return vectordb


def classify_intent(state: CodeAssistantState) -> CodeAssistantState:
    """
    Simple keyword-based intent classification:

    - If the user seems to want an explanation, set chat_state="explain".
    - Otherwise, assume generation: chat_state="generate".
    """
    text = (state.get("user_input") or "").lower()

    explain_keywords = [
        "explain", "what does", "شرح", "فسر", "explanation", "understand this code",
    ]
    if any(kw in text for kw in explain_keywords):
        intent = "explain"
    else:
        intent = "generate"

    return {"chat_state": intent}


def retrieve_code_examples(state: CodeAssistantState) -> CodeAssistantState:
    """
    Use Chroma on code examples to retrieve semantically-related snippets.
    """
    vectordb = build_code_chroma()
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    query = state.get("user_input") or ""
    docs = retriever.get_relevant_documents(query)

    examples: List[str] = []
    for d in docs:
        title = d.metadata.get("title", "example")
        examples.append(f"# {title}\n{d.page_content}")
    return {"retrieved_examples": examples}


def call_llm_for_code(state: CodeAssistantState) -> CodeAssistantState:
    """
    Call the LLM with a carefully formatted prompt, including:
    - system instructions
    - chat_state (generate or explain)
    - retrieved examples
    - user input
    """
    user_input = state.get("user_input") or ""
    chat_state = state.get("chat_state") or "generate"
    examples = state.get("retrieved_examples") or []

    SYSTEM_INSTRUCTIONS = """
You are a Python code assistant.

- If chat_state = "generate": write clean, idiomatic Python code with brief explanation.
- If chat_state = "explain": explain the given code step-by-step, then suggest improvements.
- Always produce safe, clear, and well-structured answers.
- Do not execute any external commands or access the file system.
"""

    examples_text = "\n\n".join(examples) if examples else "No examples available."

    if chat_state == "generate":
        mode_info = "User wants you to GENERATE Python code."
    else:
        mode_info = "User wants you to EXPLAIN Python code or concept."

    prompt = f"""{SYSTEM_INSTRUCTIONS}

[Routing / Intent]
chat_state = "{chat_state}"
{mode_info}

[Retrieved example snippets to inspire your answer]
{examples_text}

[User request]
{user_input}

Now respond accordingly. Start directly with the answer, no meta-commentary.
"""

    llm = get_llm()
    resp = llm.invoke(prompt)
    return {"llm_response": resp.content}


def build_code_assistant_graph():
    """
    Construct the LangGraph StateGraph for the code assistant.
    """
    graph = StateGraph(CodeAssistantState)

    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_examples", retrieve_code_examples)
    graph.add_node("call_llm", call_llm_for_code)

    graph.set_entry_point("classify_intent")
    graph.add_edge("classify_intent", "retrieve_examples")
    graph.add_edge("retrieve_examples", "call_llm")
    graph.add_edge("call_llm", END)

    return graph.compile()


@st.cache_resource(show_spinner=False)
def get_code_assistant_app():
    return build_code_assistant_graph()


# ---------------------------------------------------------------------------
# 6. STREAMLIT UI – UNIFIED SYSTEM (WEEKS 3, 4, 5)
# ---------------------------------------------------------------------------

def run_streamlit_app():
    st.set_page_config(page_title="NLP Internship Project", layout="wide")

    st.title("📚 NLP Internship – Unified LangGraph + RAG System (Weeks 3, 4 & 5)")

    # Display some global info
    st.sidebar.header("ℹ️ Info")
    st.sidebar.markdown(
        "- **Week 3**: General LLM Chat\n"
        "- **Week 4**: RAG + System Prompting + Memory + Evaluation\n"
        "- **Week 5**: LangGraph Python Code Assistant\n"
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Week 3 – General Chat",
            "Week 4 – RAG Q&A",
            "Week 5 – LangGraph Code Assistant",
            "Week 4 – RAG Evaluation & DeepEval",
        ]
    )

    # ----------------------------------------------------------------------
    # Tab 1 – General Chat (Week 3)
    # ----------------------------------------------------------------------
    with tab1:
        st.header("🗣️ General LLM Chat (Week 3)")
        st.write(
            "Ask anything related to NLP, Python, ML, or the project. "
            "This tab is a simple stateless chat interface."
        )

        user_q = st.text_area(
            "Your question:",
            height=120,
            key="w3_input",
        )

        if st.button("Send (Week 3)", key="w3_btn"):
            if not user_q.strip():
                st.warning("Please type a question.")
            else:
                try:
                    llm = get_llm()
                    resp = llm.invoke(user_q)
                    st.markdown("### Assistant")
                    st.write(resp.content)
                except Exception as e:
                    st.error(
                        f"Error calling LLM. "
                        f"Make sure you have an API key or local model configured. Details: {e}"
                    )

    # ----------------------------------------------------------------------
    # Tab 2 – RAG Q&A (Week 4)
    # ----------------------------------------------------------------------
    with tab2:
        st.header("📖 Retrieval-Augmented Q&A (Week 4)")
        st.write(
            "Ask a question about the internal knowledge base (sample docs about RAG, LangChain, etc.).\n"
            "The assistant will retrieve relevant documents and use them to answer your question."
        )

        question = st.text_area(
            "Ask a question about the documents / knowledge base:",
            height=120,
            key="w4_input",
        )

        if st.button("Ask RAG", key="w4_btn"):
            if not question.strip():
                st.warning("Please type a question.")
            else:
                with st.spinner("Running RAG chain..."):
                    try:
                        out = rag_answer(question)
                    except Exception as e:
                        st.error(f"Error running RAG chain: {e}")
                    else:
                        st.markdown("### Answer")
                        st.write(out["answer"])

                        with st.expander("🔍 View Retrieved Context"):
                            for i, c in enumerate(out["docs"], start=1):
                                st.markdown(f"**Context {i}:**")
                                st.code(c, language="markdown")

    # ----------------------------------------------------------------------
    # Tab 3 – Week 5: LangGraph Code Assistant
    # ----------------------------------------------------------------------
    with tab3:
        st.header("💻 LangGraph-powered Python Code Assistant (Week 5)")
        st.write(
            "Describe what you want in natural language. The system will:\n"
            "- Classify your intent (generate vs explain)\n"
            "- Retrieve similar code examples\n"
            "- Call the LLM with a structured prompt\n"
        )

        user_code_request = st.text_area(
            "Describe your request (e.g., 'write a function for Fibonacci' or 'explain this code: ...'):",
            height=160,
            key="w5_input",
        )

        if st.button("Run Code Assistant", key="w5_btn"):
            if not user_code_request.strip():
                st.warning("Please type a request.")
            else:
                init_state: CodeAssistantState = {"user_input": user_code_request}
                with st.spinner("Running LangGraph code assistant..."):
                    try:
                        app = get_code_assistant_app()
                        final_state = app.invoke(init_state)
                    except Exception as e:
                        st.error(f"Error running LangGraph assistant: {e}")
                    else:
                        answer = final_state.get("llm_response", "[No response]")
                        st.markdown("### Assistant")
                        st.write(answer)

                        if final_state.get("retrieved_examples"):
                            with st.expander("📦 Retrieved Example Snippets"):
                                for ex in final_state["retrieved_examples"]:
                                    st.code(ex, language="python")

    # ----------------------------------------------------------------------
    # Tab 4 – Week 4: RAG Evaluation & DeepEval
    # ----------------------------------------------------------------------
    with tab4:
        st.header("📊 RAG Evaluation & DeepEval (Week 4)")
        st.write(
            "This tab demonstrates traditional retrieval metrics (Precision@K, Recall@K, MRR, nDCG) "
            "and optionally a DeepEval metric if the library is installed."
        )

        if st.button("Run RAG Evaluation Demo"):
            metrics = demo_rag_evaluation()
            st.subheader("Demo Retrieval Metrics")
            for name, value in metrics.items():
                st.write(f"- **{name}**: `{value:.4f}`")

        st.markdown("---")
        st.subheader("DeepEval Example")

        deepeval_result = try_deepeval_example()
        if deepeval_result is None:
            st.info(
                "DeepEval is not installed. To enable this section, run:\n"
                "```bash\npip install deepeval\n```"
            )
        else:
            st.write(
                f"Metric: **{deepeval_result['metric_name']}**, "
                f"Score: `{deepeval_result['score']:.4f}`"
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Only runs when executing `python app.py` or `streamlit run app.py`
    run_streamlit_app()
