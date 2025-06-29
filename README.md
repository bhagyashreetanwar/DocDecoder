Project Description: Developed an end-to-end AI-powered application that enables intelligent conversation with multiple documents, of different formats(pdf, ppt, txt, jpg, jpeg, png, md) , simultaneously. The system converts uploaded documents into vector embeddings using FAISS (Facebook AI Similarity Search) for efficient semantic search and retrieval. Users can upload multiple research papers or documents and ask detailed questions, receiving contextually accurate responses powered by Google's Gemini Pro model integrated through LangChain framework.

Tech Stack Used:

Frontend: Streamlit (Web UI framework)

Backend & AI/ML:
Python 3.10+
Google Gemini Pro (Large Language Model)
LangChain (LLM application framework)
LangChain-Google-GenAI (Google integration)

Document Processing:
PyPDF2 (PDF text extraction)
Python-pptx (for presentations)
Pillow (for images)
Pytesseract (for image to string conversion)
RecursiveCharacterTextSplitter (Text chunking)

Vector Database & Search:
FAISS-CPU (Facebook AI Similarity Search)
Google Generative AI Embeddings (Text-to-vector conversion)

Environment & Configuration:
Python-dotenv (Environment variable management)
OS environment variables for API key management

Key Features:
Multi-document upload and processing (up to 200MB per file),
Vector-based semantic search and retrieval,
Context-aware question answering,
Local vector storage with FAISS indexing,
Modular architecture with separate functions for document processing, chunking, and vector storage

The project demonstrates proficiency in RAG (Retrieval-Augmented Generation) architecture, vector databases, and modern LLM integration patterns.
