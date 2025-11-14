---

# **Cellula – Week 5 Documentation**

### **Unified LangGraph + RAG NLP System**

### *Author: Ammar Yasser Elbedweihy*

---

## 🔵 **1. Introduction**

This document provides a semi-full technical overview of the **unified NLP system** built during Week 5 of the NLP Internship.
The system integrates work from:

* **Week 3:** General-purpose conversational LLM
* **Week 4:** RAG architecture with Chroma, Memory, System Prompting, and Evaluation
* **Week 5:** LangGraph-powered Python Code Assistant

This documentation describes:

* Overall architecture
* NLP pipeline components
* RAG and retrieval flow
* LangGraph state machine design
* Deployment approach
* Challenges encountered
* Future improvements

---

## 🔵 **2. System Overview**

The unified system merges three subsystems into one coherent application:

### **2.1 Week 3 – General LLM Chat**

A standard conversational agent capable of:

* Answering Python, ML, NLP, and general domain questions
* Maintaining natural language interaction
* Using a lightweight or external LLM backend (e.g., GPT-4o-mini)

### **2.2 Week 4 – RAG System**

Implements a **Retrieval-Augmented Generation pipeline** using:

* **Chroma** as the persistent vector database
* **Sentence Transformers** for embedding generation
* **LangChain Memory** for conversation continuity
* **ConversationalRetrievalChain** for retrieval + generation fusion

### **2.3 Week 5 – LangGraph Python Code Assistant**

A fully modular and state-driven assistant that:

* Detects user **intent** (Generate vs Explain code)
* Retrieves examples using semantic similarity search
* Passes context + examples to an LLM
* Produces structured code or explanations

The entire system is exposed through a unified **Streamlit UI**.

---

## 🔵 **3. NLP Architecture & Data Flow**

```
User Input
     ↓
Intent Classification (LangGraph)
     ↓
RAG or Code Generation Path
     ↓
Retriever → Chroma Vector Store
     ↓
Top-K Semantic Retrieval
     ↓
LLM Response Generation (Context-Aware)
     ↓
UI Output (Streamlit)
```

Each component follows standard **NLP pipeline principles**:

### ✔ **Text Preprocessing**

Raw user text → normalized + embedded using transformer models.

### ✔ **Semantic Embeddings**

Sentence Transformer: `all-MiniLM-L6-v2`

### ✔ **Vector Indexing & Similarity Search**

Chroma performs:

* Euclidean distance
* dot-product
* cosine similarity

### ✔ **Contextualized Generation**

LLM receives:

* System prompt
* Retrieved context
* User query
* Memory (summarized chat history)

---

## 🔵 **4. LangGraph State Machine Design – Week 5**

LangGraph uses a **StateGraph** to model transitions:

```
[User Input] 
     ↓
(Classify Intent)
     ↓
(Retrieve Code Examples)
     ↓
(Call LLM)
     ↓
    END
```

### **4.1 State Nodes**

| Node                | Description                                    |
| ------------------- | ---------------------------------------------- |
| `classify_intent`   | Detects Generate vs Explain intent             |
| `retrieve_examples` | Retrieves semantically similar code snippets   |
| `call_llm`          | Produces explanation or code generation output |

### **4.2 Intelligent Routing**

Keyword-based NLP intent detection:

* “explain”, “what does”, “شرح” → **explain**
* Otherwise → **generate**

This ensures the model follows the correct execution path.

---

## 🔵 **5. RAG Implementation – Week 4**

### **5.1 Embeddings**

We use transformer-based vectorization:

```python
HuggingFaceEmbeddings("sentence-transformers/all-MiniLM-L6-v2")
```

### **5.2 Chroma Vector Store**

Stores document embeddings persistently:

* Enables fast retrieval
* Survives multiple sessions (local disk persistence)

### **5.3 Memory**

`ConversationSummaryBufferMemory`

* Compresses long chat history
* Preserves semantic meaning
* Prevents context-window overflow

### **5.4 RAG Retrieval + Fusion**

Steps:

1. Retrieve `K` relevant documents
2. Rerank or filter
3. Format system + contextual prompt
4. LLM generates final grounded answer

---

## 🔵 **6. RAG Evaluation Metrics**

We implemented several retrieval quality metrics:

### **6.1 Precision@K**

Measures how many retrieved documents are relevant.

### **6.2 Recall@K**

Measures coverage of all relevant documents.

### **6.3 Mean Reciprocal Rank (MRR)**

Credit for placing the *first relevant document* high in ranking.

### **6.4 nDCG**

Discounted gain based on graded relevance scores.

These metrics help assess:

* Retrieval performance
* Relevance distribution
* Ranking quality

---

## 🔵 **7. DeepEval Integration**

DeepEval enables:

* **Answer relevancy scoring**
* **Groundedness checks**
* **Faithfulness evaluation**

It ensures that the model does not hallucinate and remains grounded in retrieved context.

---

## 🔵 **8. Streamlit Deployment**

The entire system is deployed using Streamlit with 4 main tabs:

1. **Week 3 Chat**
2. **Week 4 RAG Q&A**
3. **Week 5 LangGraph Code Assistant**
4. **RAG Evaluation + DeepEval**

Streamlit allows:

* Reactive UI
* Clean separation of components
* Rapid development

---

## 🔵 **9. Challenges Faced**

### **9.1 LLM API Consistency**

Some LLM backends frequently changed:

* Model names
* API endpoints
* Rate limits
  ⇢ Required fallback to local LLM or abstracted interfaces.

### **9.2 Vector Store Persistence Issues**

Chroma sometimes:

* Locked the database
* Required manual deletion of `.chroma` directory
* Or rebuilding embeddings

### **9.3 Memory Overflow**

Using long conversations with:

* GPT-4o-mini
* Or local LLMs

could cause:

* Token overflow
* Mismatched summary states
* Reduced answer quality

### **9.4 LangGraph Debugging**

LangGraph has:

* Complex node transitions
* Hidden execution states
* Difficult debugging for invalid states

Resolved by:

* Logging intermediate states
* Adding safe defaults
* Forcing explicit return types in TypedDict

### **9.5 Deployment Integration**

Unifying 3 weeks into *one coherent Streamlit system* required:

* Standardizing LLM interface
* Abstracting RAG chain
* Creating separate code/document vector stores
* Preventing cross-contamination of contexts

---

## 🔵 **10. Future Enhancements**

### 🚀 Add reranking models (e.g., ColBERT, Cohere Reranker)

### 🚀 Add local LLM acceleration using llama-cpp

### 🚀 Add multi-agent tools (CrewAI + LangGraph integration)

### 🚀 Add doc ingestion pipeline (PDF → text → chunks → embeddings)

### 🚀 Add user-level long-term memory

### 🚀 Add caching for repeated queries

---

## 🔵 **11. Conclusion**

The unified NLP system successfully integrates:

* **LangGraph’s deterministic state handling**
* **Chroma’s persistent vector retrieval**
* **Transformer-based embedding models**
* **LLM-guided contextual generation**
* **RAG evaluation & DeepEval verification**
* **Full Streamlit deployment**

This results in a **modular**, **scalable**, and **production-ready** Python NLP application capable of serving as:

* A code assistant
* A document QA engine
* A general-purpose LLM chat system
* A learning and experimentation platform

