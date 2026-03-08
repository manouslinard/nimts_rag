import os
import ollama
from tqdm import tqdm
from pathlib import Path
import warnings
from langchain_core._api import LangChainDeprecationWarning
warnings.simplefilter("ignore", category=LangChainDeprecationWarning)
from db_helper import DB_Helper, NimtsDocsItem
import re
import unicodedata
import time
from translatepdf import process_document

import shutil
import subprocess

def convert(file_path, target_pdf_path):
    """Function that converts word to pdf using the absolute path to LibreOffice."""
    soffice_path = r'C:\Program Files\LibreOffice\program\soffice.exe'
    output_folder = str(Path(target_pdf_path).parent)
    
    subprocess.run([
        soffice_path, 
        '--headless', 
        '--convert-to', 'pdf', 
        str(file_path), 
        '--outdir', output_folder
    ], check=True)

def convert_og_data_topdf(source='data/', dest='data_pdf/'):
    source_root = Path(source)
    output_root = Path(dest)

    output_root.mkdir(parents=True, exist_ok=True)

    files_to_process = [
        f for f in source_root.rglob("*") 
        if f.suffix.lower() in ['.docx', '.doc', '.pdf', '.txt']
        and not f.name.startswith("~$")
    ]

    for file_path in files_to_process:
        relative_path = file_path.relative_to(source_root)
        
        if file_path.suffix.lower() in ['.docx', '.doc']:
            target_path = output_root / relative_path.with_suffix('.pdf')
        else:
            target_path = output_root / relative_path

        if target_path.exists():
            print(f"Skipping (already exists): {target_path.name}")
            continue

        target_path.parent.mkdir(parents=True, exist_ok=True)

        if file_path.suffix.lower() in ['.docx', '.doc']:
            print(f"Converting: {file_path.name} -> {target_path.name}")
            try:
                convert(str(file_path), str(target_path))
            except Exception as e:
                print(f"Error converting {file_path}: {e}")
                
        elif file_path.suffix.lower() in ['.txt', '.pdf']:
            print(f"Copying: {file_path.name}")
            shutil.copy2(file_path, target_path)


def clean_text(text):
    text = unicodedata.normalize("NFKD", text)
    replacements = {
        '\u2013': '-', # en dash
        '\u2014': '-', # em dash
        '\u201c': '"', # left smart quote
        '\u201d': '"', # right smart quote
        '\u2018': "'", # left smart single quote
        '\u2019': "'"  # right smart single quote
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r'([α-ωΑ-Ωa-zA-Z])\s{2,}([α-ωΑ-Ωa-zA-Z])', r'\1\2', text)
    text = text.replace('→', '->').replace('←', '<-').replace('…', '...')

    return text.strip()

def read_txt_serially(file_path):
    file_name = Path(file_path).stem  # Gets filename without extension
    parent_folder = Path(file_path).parent.name
    content = f"DOCUMENT TITLE: {parent_folder} > {file_name}\n\n"
    content += "="*len(content) + "\n"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content += f.read()
        
    return content

from pathlib import Path

def process_entire_library(root_folder, db: DB_Helper, skip_sources=None):
    library_data = {}
    data_path = Path(root_folder)

    files_to_process = [
        f for f in data_path.rglob("*") 
        if f.suffix.lower() in ['.pdf', '.txt']
        and not f.name.startswith("~$")
    ]

    if skip_sources:
        files_to_process = [
            f for f in files_to_process
            if str(f) not in skip_sources
        ]

    processed_files = db.select_all_nimts_filenames_docs()

    files_to_process = set(str(file) for file in files_to_process) - processed_files
    # print(files_to_process)
    print(f"Found {len(files_to_process)} new documents to index.")

    def _process_any_file(path):
        ext = Path(path).suffix.lower()
        if ext == '.pdf':
            return process_document(path)
        elif ext == '.txt':
            return read_txt_serially(path)
        else:
            return None

    for file_path in processed_files:
        library_data[str(file_path)] = db.select_all_nimts_text_docs(file_path)
    
    for file_path in tqdm(files_to_process, desc='Processing pdf files...'):
        str_path = str(file_path)
        content = _process_any_file(str_path)
        if content:
            library_data[str_path] = clean_text(content)
    # print(library_data)
    return library_data


def split_save_chunks(library_data: dict, db: DB_Helper, chunk_size=800):
    for file_path, full_text in library_data.items():
        saved_chunks = db.select_all_nimts_chunkids_docs(file_path)
        
        # Αν είναι PDF, σπάσε το σε chunks
        if file_path.lower().endswith('.pdf'):
            start_idx, chunk_id = 0, 0
            while start_idx < len(full_text):
                end_idx = start_idx + chunk_size
                chunk_content = full_text[start_idx:end_idx]
                if chunk_id not in saved_chunks:
                    db.insert_nimts_docs(NimtsDocsItem(file_path, chunk_content, chunk_id))
                if end_idx >= len(full_text): break
                start_idx += chunk_size
                chunk_id += 1
        
        # Αν είναι TXT, αποθήκευσέ το όλο μαζί σαν 1 chunk (id: 0)
        elif file_path.lower().endswith('.txt'):
            if 0 not in saved_chunks:
                item = NimtsDocsItem(file_path, full_text, 0)
                db.insert_nimts_docs(item)
                db.insert_llm_summ(item)    # also inserts it as chunk explanation to the llm, since it is short.

def explain_saved_chunks(library_data:dict, db: DB_Helper):
    split_save_chunks(library_data, db)
    all_files = list(library_data.keys())
    
    base_prompt = """
    Είσαι έμπειρος τεχνικός δικτύων και αναλυτής κειμένων. 
    Σου παραθέτω ένα απόσπασμα από το έγγραφο με τίτλο: {filename}.

    Αποστολή: Προσδιόρισε το κύριο τεχνικό ζήτημα (δες και τον τίτλο) και συνοπτικά τη διαδικασία που περιγράφεται. Χρησιμοποίησε το πολύ 3 προτάσεις.

    Περιορισμοί:
    - Αν υπάρχει ήδη ΣΥΝΤΟΜΗ ΠΕΡΙΛΗΨΗ στο απόσπασμα, χρησιμοποίησε την ίδια ακριβώς ως απάντηση!
    - Χρησιμοποίησε επαγγελματικό ύφος.
    - Μην επαναλαμβάνεις την ερώτηση.

    Κείμενο προς ανάλυση:
    {text}
    """

    for filename in tqdm(all_files, desc='Creating summary for each chunk...'):
        chunk_id = 0
        explained_chunks = db.select_all_llm_chunkids_summaries(filename)
        while True:
            if chunk_id not in explained_chunks:
                text = db.select_nimts_docs(filename, chunk_id)
                if text is None:
                    break

                formatted_prompt = base_prompt.format(filename=filename, text=text)

                response = ollama.chat(
                    model="gemma3:4b-it-q4_K_M",
                    messages=[{"role": "user", "content": formatted_prompt}]
                )

                answer = response["message"]["content"]
                item = NimtsDocsItem(filename, answer, chunk_id)
                db.insert_llm_summ(item)
            chunk_id += 1

def create_total_summary(library_data:dict, db: DB_Helper):
    explain_saved_chunks(library_data, db)
    all_files = set(library_data.keys()) - db.select_total_summary_files()
    if len(all_files) == 0:
        return
    prompt_template = """
    ΡΟΛΟΣ: Είσαι ο IT Support Manager.
    ΠΛΑΙΣΙΟ: Σου παρατίθενται περιλήψεις από τμήματα (chunks) του τεχνικού αρχείου: "{filename}".
    
    ΑΠΟΣΤΟΛΗ: Συνέθεσε όλες τις πληροφορίες και γράψε ΜΙΑ (1) περιεκτική και τεχνικά ακριβή πρόταση που να 
    περιγράφει τη συνολική διαδικασία ή το πρόβλημα που πραγματεύεται το αρχείο. 
    
    ΠΕΡΙΟΡΙΣΜΟΙ:
    - Ύφος: Επαγγελματικό, τεχνικό, χωρίς περιττούς χαιρετισμούς.
    - Μην επαναλαμβάνεις την ερώτηση. Απλώς γράψε την πρόταση.

    ΔΕΔΟΜΕΝΑ (Summaries):
    {text}
    """

    for filename in tqdm(all_files, desc='Creating total summary...'):
        all_file_summaries = db.select_llm_summaries(filename)
        
        if not all_file_summaries:
            continue

        formatted_prompt = prompt_template.format(filename=filename, text=all_file_summaries)

        response = ollama.chat(
            model="gemma3:4b-it-q4_K_M",
            messages=[{"role": "user", "content": formatted_prompt}]
        )
        time.sleep(1)
        formatted_prompt = prompt_template.format(filename=filename, text=response["message"]["content"])
        response_2 = ollama.chat(
            model="gemma3:4b-it-q4_K_M",
            messages=[{"role":"user", "content": formatted_prompt}]
        )

        db.insert_total_llm_summary(filename, response_2["message"]["content"])


if __name__ == "__main__":
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"
    convert_og_data_topdf()
    time.sleep(5)
    db = DB_Helper()
    data = process_entire_library('data_pdf/', db)
    create_total_summary(data, db)
