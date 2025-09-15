# Chatbot# 📚 Conversational Chatbot with Gemini

A Streamlit-based chatbot that lets you **upload PDFs** and interact with their content using **Google Gemini** and **LangChain**.  
This app extracts text from your PDFs, splits it into chunks, stores them in a **Chroma vector database**, and enables **question-answering** with context-aware responses.  

---

## 🚀 Features
- 📂 Upload one or multiple **PDF files**.  
- 🔍 Extract and split text into **searchable chunks**.  
- 🧠 Store embeddings using **Chroma vector database**.  
- 🤖 Ask natural language questions and get context-based answers from **Google Gemini**.  
- ⚡ Powered by **LangChain** + **HuggingFace Embeddings**.  
- 🌐 Simple UI built with **Streamlit**.  

---

## 🛠️ Tech Stack
- [Streamlit](https://streamlit.io/) – Frontend web app  
- [LangChain](https://www.langchain.com/) – Retrieval & QA pipeline  
- [Chroma](https://docs.trychroma.com/) – Vector database  
- [HuggingFace Embeddings](https://huggingface.co/) – (`all-MiniLM-L6-v2`)  
- [Google Gemini](https://ai.google/) – Large Language Model  

---

## 📦 Installation

Clone the repository:
```bash
git clone https://github.com/your-username/conversational-chatbot-gemini.git
cd conversational-chatbot-gemini
