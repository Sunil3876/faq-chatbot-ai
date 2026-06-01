# 🚀 Enterprise AI FAQ Chatbot (RAG System)

An advanced, industry-level FAQ Chatbot built with **FastAPI**, **LangChain**, and **FAISS Vector Database**. Unlike traditional rule-based or basic TF-IDF bots, this system uses Semantic Search (HuggingFace Embeddings) to understand the *meaning* of user queries rather than just matching keywords.

## 🌟 Key Features
* Semantic Vector Search: Uses `sentence-transformers/all-MiniLM-L6-v2` to match context, not just keywords.
* High-Performance Database: Implements FAISS (Facebook AI Similarity Search) for lightning-fast retrieval.
* Asynchronous Backend: Powered by **FastAPI** for non-blocking, fast API responses.
* Premium UI/UX:** Built with HTML, JS, and **Tailwind CSS** featuring a dark-mode theme, typing indicators, and distance-score metrics.
* Confidence Thresholding:** Automatically detects off-topic questions if the L2 distance score exceeds 1.2.

## 🛠️ Tech Stack
* Backend: Python, FastAPI, Uvicorn
* AI/NLP:LangChain, HuggingFace Embeddings
* Database: FAISS (In-Memory Vector Store)
* Frontend: HTML5, JavaScript, Tailwind CSS (via CDN)

## ⚙️ Installation & Setup

1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/faq-chatbot-ai.git](https://github.com/yourusername/faq-chatbot-ai.git)
   cd faq-chatbot-ai
   Install required dependencies:

2. Install required dependencies: pip install -r requirements.txt
3. Run the FastAPI server: python main.py
4.Access the application: http://127.0.0.1:8000
