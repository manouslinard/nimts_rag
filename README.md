# NIMTSgpt: Local IT Support RAG System

NIMTSgpt is a specialized Retrieval-Augmented Generation (RAG) pipeline designed to process technical IT support documentation and provide intelligent, context-aware answers. It features an automated document conversion engine and a local LLM backend via Ollama.

Database with guides for nimts it support has not been uploaded on github, due to privacy reasons.

## 🚀 Features
* **Multi-Format Processing:** Automatically converts legacy `.doc` and `.docx` files to PDF using LibreOffice.
* **Deep Document Analysis:** Uses `Docling` for high-fidelity PDF parsing. Images are parsed and explained by a local VLM.
* **Local & Private:** Powered by `Ollama` (Gemma 3) to ensure all data stays on your local machine.
* **Smart Summarization:** Generates both chunk-level and document-wide summaries for better retrieval.
* **Web Interface:** A FastAPI-based web UI for real-time chatting with your document library.

## 🛠️ Prerequisites
1.  **LibreOffice:** Required for Word-to-PDF conversion. Ensure `soffice` is in your system PATH.
2.  **Ollama:** Install [Ollama](https://ollama.ai/) and pull the quantisized model:
    ```bash
    ollama pull gemma4:e2b-it-q4_K_M
    ```

## 📂 Project Structure
* `populate_db.py`: The ingestion engine. Converts docs, cleans text, and populates the SQLite database.
* `query_db.py`: The retrieval logic using Cosine Similarity to find relevant document chunks.
* `app.py`: FastAPI web server and frontend implementation.
* `translatepdf.py`: Advanced PDF parsing using the Docling framework.
* `db_helper.py`: Database schema and SQLAlchemy operations.

## ⚙️ Setup & Usage

### 1. Ingest Data
Place your raw files in the `/data` folder, then run:
```bash
python populate_db.py
```

This will create a `/data_pdf` folder with converted files and a nimts_rag SQLite database.

### 2. Run the Web UI
```bash
python app.py
```

Navigate to [http://localhost:5000](http://localhost:5000) to start asking questions.

#### ❓Example questions:
1. Πως μπορω να κανω reset password σε εναν χρηστη?
2. Ο μεγάλος konica στα εργαστηρια δεν εκτυπωνει. Πως το λυνω αυτο?
3. Δεν μπορω να εκτυπωσω στον konica στο μικροβιολογικο. Πως λυνεται αυτο?
4. Εχω κοκκινη οθονη. Τι μπορω να κανω?
5. Πως κανω activate τα windows?
6. Πως φτιαχνω τις ρυθμισεις του zebra barcode?
