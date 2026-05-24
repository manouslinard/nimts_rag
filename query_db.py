import os
import re
import argparse
import asyncio
import urllib.parse
from pathlib import Path
import numpy as np
from ollama import AsyncClient
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi

from db_helper import DB_Helper

# ==========================================
# 1. DEFINE GLOBALS (Empty at start)
# ==========================================
db_helper = None
filenames = []
embedding_model = None
corpus_embeddings = None
bm25 = None

def tokenize_greek(text):
    """Βοηθητική συνάρτηση για το BM25 Tokenization"""
    if not text: return []
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.split()

def initialize_hybrid_retriever():
    global db_helper, filenames, embedding_model, corpus_embeddings, bm25
    
    print("Initializing Hybrid Retriever (Loading Models & Data)...")
    db_helper = DB_Helper()
    answer_dict = db_helper.select_total_llm_allfiles()

    filenames = list(answer_dict.keys()) if answer_dict else []
    summaries = [answer_dict[f] for f in filenames]
    full_texts = [db_helper.select_all_nimts_text_docs(f) for f in filenames]

    tokenized_corpus = [tokenize_greek(doc) for doc in full_texts]
    bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None

    embedding_model = SentenceTransformer('intfloat/multilingual-e5-small')
    if summaries:
        corpus_embeddings = embedding_model.encode(["passage: " + s for s in summaries], convert_to_tensor=True)

    print("Hybrid Retriever is ready!")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def make_clickable_link(path, base_url=None):
    web_path = path.replace("\\", "/")
    display_path = Path(path).stem
    if base_url:
        safe_path = urllib.parse.quote(web_path)
        url = f"http://{base_url.rstrip('/')}/{safe_path}"
        return f'<a href="{url}" target="_blank" style="color: #007bff; text-decoration: underline;">{display_path}</a>'
    else:
        abs_path = os.path.abspath(path).replace("\\", "/")
        safe_path = urllib.parse.quote(abs_path)
        url = f"file:///{safe_path}"
        return f"\033]8;;{url}\033\\{path}\033]8;;\033\\"

async def question_db(user_query, k=3):
    if not filenames:
        return "Δεν βρέθηκαν έγγραφα στη βάση.", []

    query_embedding = embedding_model.encode("query: " + user_query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    vector_ranking = np.argsort(cos_scores.cpu().numpy())[::-1]

    tokenized_query = tokenize_greek(user_query)
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_ranking = np.argsort(bm25_scores)[::-1]

    rrf_k = 60
    rrf_scores = {i: 0.0 for i in range(len(filenames))}

    for rank, doc_idx in enumerate(vector_ranking):
        rrf_scores[doc_idx] += 1.0 / (rrf_k + rank + 1)
        
    for rank, doc_idx in enumerate(bm25_ranking):
        rrf_scores[doc_idx] += 1.0 / (rrf_k + rank + 1)

    sorted_indices = sorted(rrf_scores.keys(), key=lambda idx: rrf_scores[idx], reverse=True)
    top_k_indices = sorted_indices[:k]
    
    best_file_paths = [filenames[idx] for idx in top_k_indices]
    
    # --- ΔΙΟΡΘΩΣΗ: XML Formatting & Truncation ---
    combined_context = ""
    for i, path in enumerate(best_file_paths):
        doc_text = db_helper.select_all_nimts_text_docs(path)
        
        # Περιορίζουμε το μέγεθος κάθε εγγράφου στους ~6000 χαρακτήρες 
        # για να μην κρασάρει η μνήμη του μικρού μοντέλου
        if len(doc_text) > 6000:
            doc_text = doc_text[:6000] + "\n...[Το κείμενο συνεχίζεται στο πρωτότυπο αρχείο]..."
            
        # Χρήση XML tags για να αναγκάσουμε το LLM να το δει ως ξεχωριστή οντότητα
        combined_context += f'<document id="{i+1}" title="{Path(path).stem}">\n{doc_text}\n</document>\n\n'
    
    return combined_context, best_file_paths

# --- ΝΕΕΣ ΣΥΝΑΡΤΗΣΕΙΣ ΓΙΑ ΤΟ 2-STEP UI ---

async def get_documents(user_query, base_server_url: str = None):
    """Βήμα 1: Επιστρέφει ακαριαία τα σχετικά έγγραφα"""
    result = await question_db(user_query)
    context, ranks = result[0], result[1]
    
    links_section = "Σχετικά έγγραφα:\n"
    for i, file_path in enumerate(ranks, 1):
        clickable = make_clickable_link(file_path, base_server_url)
        links_section += f"{i}. {clickable}\n"
        
    return context, links_section

async def ask_llm(user_query, context):
    """Βήμα 2: Αναλαμβάνει το βαρύ κομμάτι του LLM"""
    client = AsyncClient()
    # print(context)
    # --- ΔΙΟΡΘΩΣΗ: Πιο αυστηρό Prompt ---
    prompt = f"""Είσαι ένας IT Expert. Πρέπει να απαντήσεις στην ερώτηση του χρήστη με βάση τα παρακάτω έγγραφα.

    ΔΕΔΟΜΕΝΑ (Μέσα σε XML tags):
    {context}

    ΕΡΩΤΗΣΗ ΧΡΗΣΤΗ: 
    {clean_text(user_query)}

    ΑΥΣΤΗΡΕΣ ΟΔΗΓΙΕΣ:
    1. Διάβασε το σχετικό έγγραφο μέχρι το ΤΕΛΟΣ.
    2. Κατέγραψε ΟΛΑ τα βήματα της διαδικασίας, χωρίς να παραλείψεις κανένα. ΜΗΝ σταματάς στη μέση.
    3. Αν το έγγραφο αναφέρει εντολές τερματικού (terminal commands) ή κώδικα (π.χ. disable, enable, ipstat), ΠΡΕΠΕΙ υποχρεωτικά να τις γράψεις και αυτές ακριβώς όπως είναι.
    4. Δώσε κατευθείαν τη λύση με βήματα στα ελληνικά.
    """

    response = await client.chat(
        model="gemma4:e2b-it-q4_K_M",
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": 0, 
            "seed": 42,
            "num_ctx": 8192  
        }
    )
    
    return response['message']['content'].strip()

# Αν τρέξει αυτόνομα στο τερματικό
if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--query", type=str, help="The question to ask.")
    args = parser.parse_args()

    initialize_hybrid_retriever()
    if args.query:
        # Simulator για το τερματικό
        async def terminal_sim():
            ctx, links = await get_documents(args.query)
            print(f"\n{links}\n...Generating AI response...\n")
            ans = await ask_llm(args.query, ctx)
            print(f"Απάντηση:\n{ans}")
        asyncio.run(terminal_sim())