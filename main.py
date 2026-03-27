from fastapi import FastAPI
from pydantic import BaseModel
from decision_engine import DecisionEngine
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI(title="Nivetix LLM-Free AI Chatbot Framework", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = DecisionEngine()

# In-Memory Cache for Session History contexts
session_history = {}

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default_user_session"

class ChatResponse(BaseModel):
    response: str

@app.get("/health")
def health_check():
    return {
        "status": "online", 
        "engine": "Hybrid Scikit + FAISS", 
        "llm_dependency": "FALSE"
    }

@app.get("/")
@app.get("/chat")
def browser_test_ui():
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html lang="en">
    <body style="font-family: system-ui, sans-serif; background: #0a0a0a; color: #fff; padding: 3rem; max-width: 800px; margin: 0 auto;">
        <h2 style="color: #a855f7;">🤖 Nivetix LLM-Free Engine Live!</h2>
        <p style="color: #a1a1aa;">You tried to open this URL directly in the browser, which triggered a <b>GET</b> request. The chatbot securely accepts <b>POST</b> requests formatted in JSON.</p>
        
        <div style="background: #18181b; padding: 2rem; border-radius: 12px; margin-top: 2rem; border: 1px solid #27272a;">
            <h3 style="margin-top: 0;">Test the Bot Right Now:</h3>
            <div style="display: flex; gap: 10px;">
                <input id="queryBox" type="text" placeholder="Ask something (e.g. 'websit kitne ka?')" style="flex: 1; padding: 12px; border-radius: 8px; border: 1px solid #3f3f46; background: #27272a; color: white;" />
                <button onclick="fireChat()" style="background: #a855f7; color: white; border: none; padding: 0 24px; border-radius: 8px; cursor: pointer; font-weight: bold;">Send POST Request</button>
            </div>
            
            <div id="responseBox" style="margin-top: 2rem; padding: 1.5rem; background: #000; border-radius: 8px; border: 1px solid #27272a; font-family: monospace; white-space: pre-wrap; min-height: 80px; color: #4ade80;">
                Awaiting input...
            </div>
        </div>

        <script>
            async function fireChat() {
                const text = document.getElementById('queryBox').value;
                if(!text) return;
                
                const rb = document.getElementById('responseBox');
                rb.style.color = '#a1a1aa';
                rb.innerText = "Transmitting POST request to /chat...";
                
                try {
                    const res = await fetch('/chat', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({ message: text, session_id: "browser_test_user" })
                    });
                    const data = await res.json();
                    rb.style.color = '#4ade80';
                    rb.innerText = data.response;
                } catch(e) {
                    rb.style.color = '#ef4444';
                    rb.innerText = "Error: " + e;
                }
            }
        </script>
    </body>
    </html>
    """)

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # Retrieve contextual history
    history = session_history.get(req.session_id, [])
    
    # Process payload utilizing dual-processor engine
    response_text = await engine.process_message(req.message, history)
    
    # Commit interaction to contextual memory constraint cache
    history.append(req.message)
    if len(history) > 2:
        history = history[-2:]
    session_history[req.session_id] = history
    
    return ChatResponse(response=response_text)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
