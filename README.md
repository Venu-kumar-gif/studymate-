# 📘 PDF Question Answering System

This project is an **AI-powered Question Answering system** that allows users to upload a PDF file and ask questions.  
The system retrieves relevant chunks of text from the document and generates accurate answers using **NLP models and FAISS vector search**.

---

## 🚀 Features
- Upload and process **PDF documents**.
- Ask **questions in natural language** about the PDF.
- Uses **embeddings + FAISS** for efficient search.
- Generates **context-aware answers**.
- Simple **Streamlit web interface**.

---

## 🛠️ Tech Stack
- **Python 3.9+**
- **Streamlit** – Web app UI
- **PyMuPDF (fitz)** – PDF text extraction
- **Sentence-Transformers** – Text embeddings
- **FAISS** – Vector database for similarity search
- **Hugging Face Transformers** – FLAN-T5 model for Q&A
- **PyTorch** – Backend for models

---

## ⚡ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/pdf-qa-system.git
   cd pdf-qa-system
   ```
   ---

2. Create a virtual environment
      ```
      python -m venv venv
      source venv/bin/activate   # Linux/Mac
      venv\Scripts\activate      # Windows
      ```

  --- 

3. Install dependencies
     ```
      pip install -r requirements.txt 
     ```

  ---

4. Run the Streamlit app
     ```
     streamlit run app.py
     ```

  ---
  
5. Project Structure  
     ```
     pdf-qa-system/
      │-- app.py                # Main Streamlit application
      │-- requirements.txt      # List of dependencies
      │-- README.md             # Documentation
      │-- /venv                 # Virtual environment (ignored in git)
      ```  
      ---
