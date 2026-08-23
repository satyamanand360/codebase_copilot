# Codebase Copilot 🚀

An AI-powered codebase assistant utilizing a robust RAG (Retrieval-Augmented Generation) architecture to securely answer developer questions by retrieving, synthesizing, and reasoning over local code chunks.

## 🌟 Features
- **Hybrid Search**: Combines Dense Vector Similarity with Sparse Keyword matching (via Qdrant) for high-recall code retrieval.
- **Cross-Encoder Reranking**: Boosts context precision by re-evaluating retrieved chunks against the exact query context.
- **Multi-Hop Graph Expansion**: Extracts implicit dependencies, imports, and symbol references to provide deep contextual understanding.
- **Automated Benchmarking**: Includes a custom RAGAS-style LLM-as-a-judge pipeline to continually evaluate hallucinations and accuracy.

## 🛠️ Architecture
- **Orchestration**: LangChain
- **Vector Database**: Local Qdrant Engine
- **Embedding Model**: all-MiniLM-L6-v2 (HuggingFace)
- **Generation & Judge LLM**: Google Gemini 3.1 Flash Lite

## 📊 Evaluation & Metrics
Evaluated using an automated LLM-as-a-judge pipeline measuring strict retrieval and hallucination boundaries.

- **Faithfulness (Anti-hallucination)**: 50.0%
- **Context Precision (Retrieval accuracy)**: 50.0%
- **Context Recall (Retrieval coverage)**: 50.0%

> *Metrics achieved on a targeted codebase benchmark evaluated by Gemini-as-a-judge.*

## 🚀 Quick Start

`ash
# 1. Activate the virtual environment
.\venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API Keys
# Create a .env file and add your GOOGLE_API_KEY

# 4. Run the Streamlit Dashboard
streamlit run app.py
`
