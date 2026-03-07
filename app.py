import os
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

from query_db import query_db

app = FastAPI()

if os.path.exists("data_pdf"):
    app.mount("/data_pdf", StaticFiles(directory="data_pdf"), name="data_pdf")

SERVER_URL = "0.0.0.0"
PORT = 5000

class Question(BaseModel):
    question: str

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NIMTSgpt</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            background: #f4f6f9;
            padding: 15px; /* Added padding for mobile edges */
        }

        h2 {
            text-align: center;
            color: #333;
            margin-top: 10px;
        }

        #chat {
            background: white;
            padding: 15px;
            border-radius: 12px;
            height: 60vh; /* Responsive height based on screen */
            min-height: 300px;
            overflow-y: auto;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            border: 1px solid #ddd;
        }

        .row {
            display: flex;
            margin: 12px 0;
        }

        .row.user { justify-content: flex-end; }
        .row.bot { justify-content: flex-start; }

        .msg {
            padding: 10px 14px;
            border-radius: 15px;
            max-width: 85%; /* Slightly wider for mobile */
            font-size: 15px;
            line-height: 1.4;
        }

        .user .msg {
            background: #007bff;
            color: white;
            border-bottom-right-radius: 2px;
        }

        .bot .msg {
            background: #e9ecef;
            color: #333;
            border-bottom-left-radius: 2px;
            white-space: pre-wrap;
        }

        .controls {
            display: flex;
            gap: 8px;
            margin-top: 15px;
        }

        input {
            flex: 1;
            padding: 12px;
            font-size: 16px; /* 16px prevents iOS zoom-on-focus */
            border-radius: 8px;
            border: 1px solid #ccc;
            outline: none;
        }

        button {
            padding: 12px 15px;
            font-size: 15px;
            border-radius: 8px;
            border: none;
            cursor: pointer;
            font-weight: 600;
            transition: background 0.2s;
        }

        .send-btn { background: #28a745; color: white; }
        .clear-btn { background: #dc3545; color: white; }

        /* MOBILE ADJUSTMENTS */
        @media (max-width: 600px) {
            body { margin: 10px auto; }
            
            .controls {
                flex-wrap: wrap; /* Allows buttons to drop to next line if needed */
            }

            input {
                width: 100%;
                flex: none; /* Input takes full top row */
            }

            .send-btn {
                flex: 2; /* Send button is wider */
            }

            .clear-btn {
                flex: 1; /* Clear button is smaller */
                font-size: 12px;
            }
        }

        .loading:after {
            content: '.';
            animation: dots 1s steps(5, end) infinite;
            }

        @keyframes dots {
            0%, 20% { content: '.'; }
            40% { content: '..'; }
            60% { content: '...'; }
            80%, 100% { content: ''; }
        }

        .loading-row {
            font-style: italic;
            color: #666;
            margin: 10px 0;
        }        
    </style>
</head>
<body>
    <h2>NIMTSgpt</h2>

    <div id="chat"></div>

    <div class="controls">
        <input id="question" type="text" placeholder="Ask a question..." onkeydown="handleEnter(event)" />
        <button class="send-btn" onclick="send()">Send</button>
        <button class="clear-btn" onclick="clearChat()">Clear</button>
    </div>

    <script>
        // Define these once at the top
        const chat = document.getElementById("chat");
        const input = document.getElementById("question"); // Changed from "input" to "question"

        async function send() {
            const text = input.value.trim();
            if (!text) return;

            // 1. Show user message
            chat.innerHTML += `<div class="row user"><div class="msg">${text}</div></div>`;
            input.value = "";
            
            // 2. Add the loading indicator
            const loadingId = "loading-" + Date.now();
            chat.innerHTML += `
                <div class="row loading-row" id="${loadingId}">
                    Generating LLM response<span class="loading"></span>
                </div>`;
            chat.scrollTop = chat.scrollHeight;

            try {
                const response = await fetch("/ask", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ question: text })
                });
                const data = await response.json();

                // 3. Remove the loading indicator
                const loadingElement = document.getElementById(loadingId);
                if (loadingElement) loadingElement.remove();

                // 4. Show bot message
                chat.innerHTML += `
                    <div class="row bot">
                        <div class="msg">${data.answer}</div>
                    </div>`;
            } catch (e) {
                const loadingElement = document.getElementById(loadingId);
                if (loadingElement) loadingElement.remove();
                chat.innerHTML += `<div class="row bot"><div class="msg" style="color:red">Error connecting to server.</div></div>`;
            }
            chat.scrollTop = chat.scrollHeight;
        }

        async function clearChat() {
            chat.innerHTML = "";
        }

        function handleEnter(event) {
            if (event.key === "Enter") {
                send();
            }
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def home():
    # clear_history()
    return HTML

@app.post("/ask")
async def ask(data: Question, request: Request):
    host_ip = request.url.hostname 
    server = f"{host_ip}:{PORT}"
    answer = await query_db(user_query=data.question, base_server_url=server)
    return {"answer": answer}

# @app.post("/clear")
# async def clear():
#     clear_history()
#     return {"status": "cleared"}

@app.get('/prompt')
async def question_llm(user_query: str):
    return {'answer': query_db(user_query=user_query)}

if __name__ == "__main__":
    uvicorn.run("app:app", host=str(SERVER_URL), port=int(PORT), reload=True)
