# NIMTSgpt: Local IT Support RAG System

NIMTSgpt is a specialized Retrieval-Augmented Generation (RAG) pipeline designed to process technical IT support documentation and provide intelligent, context-aware answers. It features an automated document conversion engine and a local LLM backend via Ollama.

Database with guides for nimts it support has not been uploaded on github, due to privacy reasons.

## 🚀 Features
* **Multi-Format Processing:** Automatically converts legacy `.doc` and `.docx` files to PDF using LibreOffice.
* **Deep Document Analysis:** Uses `Docling` for high-fidelity PDF parsing. Images are parsed and explained by a local VLM.
* **Hybrid Search:** Combines BM25 keyword search with `intfloat/multilingual-e5-small` embeddings, merged via Reciprocal Rank Fusion (RRF).
* **Local & Private:** Powered by `Ollama` (Gemma 3 / Gemma 4) so all data stays on your machine.
* **Smart Summarization:** Generates both chunk-level and document-wide summaries for better retrieval.
* **Web Interface:** A FastAPI-based web UI for real-time chatting with your document library.

## 🛠️ Prerequisites
1.  **LibreOffice:** Required for Word-to-PDF conversion. `populate_db.py` expects it at `C:\Program Files\LibreOffice\program\soffice.exe` (Windows); edit `convert()` for other setups.
2.  **Ollama:** Install [Ollama](https://ollama.ai/) and pull both quantized models:
    ```bash
    ollama pull gemma3:4b-it-q4_K_M   # ingestion: summaries + image (VLM) descriptions
    ollama pull gemma4:e2b-it-q4_K_M  # answering questions in the web UI
    ```
3.  **Python deps:** `pip install -r requirements.txt`. On first run, `intfloat/multilingual-e5-small` is downloaded from Hugging Face (internet needed once).

## 📂 Project Structure
* `populate_db.py`: The ingestion engine. Converts docs, cleans text, and populates the SQLite database.
* `query_db.py`: The hybrid retrieval logic (BM25 + embeddings, fused with RRF) and the LLM answer generation.
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

> **Note:** the server binds to `0.0.0.0`, so it is reachable from other machines on your network. Set `SERVER_URL = "127.0.0.1"` in `app.py` for local-only access.

#### ❓Example questions:
1. Πως μπορω να κανω reset password σε εναν χρηστη?
2. Ο εκτυπωτής στο εργαστήριο δεν εκτυπώνει. Πως το λυνω αυτο?
3. Δεν μπορω να εκτυπωσω στον εκτυπωτή του τμήματος. Πως λυνεται αυτο?
4. Εχω κοκκινη οθονη. Τι μπορω να κανω?
5. Πως κανω activate τα windows?
6. Πως φτιαχνω τις ρυθμισεις του εκτυπωτή barcode?
