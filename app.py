import os
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

# Εισάγουμε τις νέες συναρτήσεις
from query_db import get_documents, ask_llm, initialize_hybrid_retriever

app = FastAPI()

if os.path.exists("data_pdf"):
    app.mount("/data_pdf", StaticFiles(directory="data_pdf"), name="data_pdf")

@app.on_event("startup")
async def startup_event():
    initialize_hybrid_retriever()

SERVER_URL = "0.0.0.0"
PORT = 5000

# Models για το API
class QuestionRequest(BaseModel):
    question: str

class GenerateRequest(BaseModel):
    question: str
    context: str

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NIMTSgpt</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; max-width: 900px; margin: 0 auto; background: #f4f6f9; padding: 15px; }
        h2 { text-align: center; color: #333; margin-top: 10px; }
        #chat { background: white; padding: 15px; border-radius: 12px; height: 60vh; min-height: 300px; overflow-y: auto; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border: 1px solid #ddd; }
        .row { display: flex; margin: 12px 0; }
        .row.user { justify-content: flex-end; }
        .row.bot { justify-content: flex-start; }
        .msg { padding: 10px 14px; border-radius: 15px; max-width: 85%; font-size: 15px; line-height: 1.4; }
        .user .msg { background: #007bff; color: white; border-bottom-right-radius: 2px; }
        .bot .msg { background: #e9ecef; color: #333; border-bottom-left-radius: 2px; white-space: pre-wrap; }
        .controls { display: flex; gap: 8px; margin-top: 15px; }
        input { flex: 1; padding: 12px; font-size: 16px; border-radius: 8px; border: 1px solid #ccc; outline: none; }
        button { padding: 12px 15px; font-size: 15px; border-radius: 8px; border: none; cursor: pointer; font-weight: 600; transition: background 0.2s; }
        .send-btn { background: #28a745; color: white; }
        .clear-btn { background: #dc3545; color: white; }
        @media (max-width: 600px) { body { margin: 10px auto; } .controls { flex-wrap: wrap; } input { width: 100%; flex: none; } .send-btn { flex: 2; } .clear-btn { flex: 1; font-size: 12px; } }
        
        .loading:after { content: '.'; animation: dots 1.5s steps(5, end) infinite; }
        @keyframes dots { 0%, 20% { content: '.'; } 40% { content: '..'; } 60% { content: '...'; } 80%, 100% { content: ''; } }
        .loading-text { font-style: italic; color: #666; margin: 5px 0; }
    </style>
</head>
<body>
    <h2>NIMTSgpt</h2>
    <div id="chat"></div>
    <div class="controls">
        <input id="question" type="text" placeholder="Κάνε μια ερώτηση..." onkeydown="handleEnter(event)" />
        <button class="send-btn" onclick="send()">Send</button>
        <button class="clear-btn" onclick="clearChat()">Clear</button>
    </div>

    <script>
        const chat = document.getElementById("chat");
        const input = document.getElementById("question");

        async function send() {
            const text = input.value.trim();
            if (!text) return;

            // 1. Εμφάνιση ερώτησης χρήστη
            chat.innerHTML += `<div class="row user"><div class="msg">${text}</div></div>`;
            input.value = "";
            
            // Δημιουργία ID για το συγκεκριμένο μήνυμα του bot ώστε να το ανανεώνουμε
            const botMsgId = "bot-" + Date.now();
            
            // Αρχικό state: Ψάχνει έγγραφα
            chat.innerHTML += `
                <div class="row bot">
                    <div class="msg" id="${botMsgId}">
                        <div class="loading-text">Αναζήτηση σχετικών εγγράφων<span class="loading"></span></div>
                    </div>
                </div>`;
            chat.scrollTop = chat.scrollHeight;

            try {
                // ΒΗΜΑ 1: Ζητάμε τα Έγγραφα (Γρήγορο)
                const searchResponse = await fetch("/search", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ question: text })
                });
                const searchData = await searchResponse.json();

                // Ενημερώνουμε το UI: Δείχνουμε τα έγγραφα και βάζουμε το AI να φορτώνει
                const botBox = document.getElementById(botMsgId);
                botBox.innerHTML = searchData.links + 
                    `\\n======================\\n\\n` + 
                    `<div class="loading-text">Ο AI Βοηθός συντάσσει απάντηση<span class="loading"></span></div>`;
                chat.scrollTop = chat.scrollHeight;

                // ΒΗΜΑ 2: Ζητάμε την απάντηση του LLM (Αργό)
                const llmResponse = await fetch("/generate", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ 
                        question: text, 
                        context: searchData.context 
                    })
                });
                const llmData = await llmResponse.json();

                // Τελικό Update UI: Κρατάμε τα links και προσθέτουμε την τελική απάντηση
                botBox.innerHTML = searchData.links + 
                    `\\n======================\\n\\n` + 
                    `<b>Συνοπτική απάντηση ΑΙ βοηθού:</b>\\n` + llmData.answer;
                
            } catch (e) {
                const botBox = document.getElementById(botMsgId);
                if (botBox) {
                    botBox.innerHTML = `<span style="color:red">Σφάλμα επικοινωνίας με τον διακομιστή.</span>`;
                }
            }
            chat.scrollTop = chat.scrollHeight;
        }

        function clearChat() { chat.innerHTML = ""; }
        function handleEnter(event) { if (event.key === "Enter") send(); }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML

# ΝΕΟ ENDPOINT 1: Φέρνει ακαριαία τα έγγραφα
@app.post("/search")
async def search_endpoint(data: QuestionRequest, request: Request):
    host_ip = request.url.hostname 
    server = f"{host_ip}:{PORT}"
    
    # Παίρνουμε links και κείμενο
    context, links = await get_documents(user_query=data.question, base_server_url=server)
    
    # Επιστρέφουμε τα links για εμφάνιση και το context για να το στείλει πίσω το frontend
    return {"links": links, "context": context}

# ΝΕΟ ENDPOINT 2: Γράφει την απάντηση του LLM
@app.post("/generate")
async def generate_endpoint(data: GenerateRequest):
    # Δίνουμε το έτοιμο context στο LLM, γλιτώνοντας χρόνο
    answer = await ask_llm(user_query=data.question, context=data.context)
    return {"answer": answer}

if __name__ == "__main__":
    uvicorn.run("app:app", host=str(SERVER_URL), port=int(PORT), reload=True)