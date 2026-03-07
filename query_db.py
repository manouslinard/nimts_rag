import os
import argparse
import ollama
from db_helper import DB_Helper
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import urllib.parse
from pathlib import Path
import urllib.parse

def make_clickable_link(path, base_url=None):
    web_path = path.replace("\\", "/")
    display_path = Path(path).stem
    if base_url:
        # WEB VERSION: Standard HTML link
        safe_path = urllib.parse.quote(web_path)
        url = f"http://{base_url.rstrip('/')}/{safe_path}"
        return f'<a href="{url}" target="_blank" style="color: #007bff; text-decoration: underline;">{display_path}</a>'
    else:
        # TERMINAL VERSION: OSC 8
        abs_path = os.path.abspath(path).replace("\\", "/")
        safe_path = urllib.parse.quote(abs_path)
        url = f"file:///{safe_path}"
        return f"\033]8;;{url}\033\\{path}\033]8;;\033\\"

def find_most_similar(x, string_set):
    all_strings = [x] + list(string_set)    
    vectorizer = CountVectorizer().fit_transform(all_strings)
    vectors = vectorizer.toarray()
    similarities = cosine_similarity(vectors[0:1], vectors[1:])
    most_similar_index = similarities.argmax()
    return list(string_set)[most_similar_index]

def question_db(user_query, k=3):
    db_helper = DB_Helper()
    answer_dict = db_helper.select_total_llm_allfiles()     

    s = ""
    for i, (key, value) in enumerate(answer_dict.items()):
        s += f"ΤΙΤΛΟΣ ΕΓΓΡΑΦΟΥ #{i}: {os.path.basename(key)}\nΠΕΡΙΓΡΑΦΗ: {value}\n----------\n"

    system_instruction = (
        "Είσαι ένας έμπειρος τεχνικός IT. Ανάλυσε ΠΡΟΣΕΚΤΙΚΑ τη λίστα εγγράφων που θα σου δοθεί. "
        "Πρέπει να βρεις τους τίτλους που σχετίζονται ΑΚΡΙΒΩΣ με το πρόβλημα του χρήστη. "
        "Απάντησε ΑΠΟΚΛΕΙΣΤΙΚΑ σε JSON μορφή χωρίς άλλο κείμενο."
    )

    user_content = f"""ΕΡΩΤΗΣΗ ΧΡΗΣΤΗ: "{user_query}"

    ΛΙΣΤΑ ΕΓΓΡΑΦΩΝ ΠΡΟΣ ΑΝΑΛΥΣΗ:
    {s}

    ΟΔΗΓΙΑ: Βρες τους {k} πιο σχετικούς τίτλους. 
    Απάντησε σε JSON: {{ "1": "Τίτλος 1", "2": "Τίτλος 2" ... }}"""

    response = ollama.chat(
        model="gemma3:4b-it-q4_K_M",
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_content}
        ],
        format="json",
        options={
            "temperature": 0,
            "seed": 42,
            "num_ctx": 16384
        }
    )

    answer_raw = response['message']['content']
    all_files = db_helper.select_all_nimts_filenames_docs()

    best_file_paths = []
    for i in range(1, k+1):
        best_file_paths.append(find_most_similar(str(json.loads(answer_raw)[str(i)]), all_files))
    
    best_match = best_file_paths[0]
    # print(best_match)
    # returns the most relevant guide (all of it):
    return db_helper.select_all_nimts_text_docs(best_match), best_file_paths

# HISTORY_FILE = "chat_history.json"

# def load_history():
#     if os.path.exists(HISTORY_FILE):
#         with open(HISTORY_FILE, "r") as f:
#             return json.load(f)
#     return []

# def save_history(history):
#     with open(HISTORY_FILE, "w") as f:
#         json.dump(history[-6:], f)

def query_db(user_query, base_server_url: str = None):
    # history = load_history()
    result = question_db(user_query)
    context, ranks = result[0], result[1]

    prompt = f"""
        Είσαι ένας έμπειρος τεχνικός υποστήριξης IT. 
        Χρησιμοποίησε ΑΠΟΚΛΕΙΣΤΙΚΑ τις παρακάτω πληροφορίες για να δώσεις μια ΑΜΕΣΗ και ΑΝΑΛΥΤΙΚΗ λύση στον χρήστη.
        
        ΚΑΝΟΝΕΣ:
        1. ΜΗΝ λες στον χρήστη "διαβάστε το αρχείο" ή "ακολουθήστε τα βήματα στο κείμενο".
        2. Εξήγησε τα βήματα ΕΔΩ στην απάντηση σου με bullet points.
        3. Απάντησε στα ελληνικά.
    
        Περιεχόμενο Εγγράφου:
        {context}

        Ερώτηση Χρήστη:
        {user_query}
    """

    # messages = history + [{"role": "user", "content": prompt}]
    messages = [{"role": "user", "content": prompt}]

    response = ollama.chat(
        model="gemma3:4b-it-q4_K_M",   # larger context window.
        messages=messages,
        options={
            "temperature": 0, # Μηδενικό temperature για deterministic απαντήσεις
            "seed": 42        # Το ίδιο seed για σταθερότητα
        }
    )
    
    links_section = "Σχετικά έγγραφα:\n"
    for i, file_path in enumerate(ranks, 1):
        clickable = make_clickable_link(file_path, base_server_url)
        links_section += f"{i}. {clickable}\n"

    answer = f"\n{links_section}"
    answer += '=' * 15 + '\n\n' + f"Συνοπτική απάντηση ΑΙ βοηθού:\n{response['message']['content']}".strip()

    # history.append({"role": "user", "content": user_query})
    # history.append({"role": "assistant", "content": answer})
    # save_history(history)

    print(f"\nAssistant: {answer}")
    return answer

# def clear_history():
#     if os.path.exists(HISTORY_FILE):
#         os.remove(HISTORY_FILE)
#         print("History cleared.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--query", type=str, help="The question to ask.")
    # parser.add_argument("--clear", action="store_true", help="Clear chat history.")
    
    args = parser.parse_args()

    # if args.clear:
    #     clear_history()
    if args.query:
        query_db(args.query)
