import os
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.jarvis import Jarvis
from core.intent_router import route
from memory.recall import get_recent_conversations, search_memory
from config.settings import settings

app = FastAPI(
    title="JARVIS AI Autonomous Desktop Agent API",
    version="1.0.0",
    description="Voice-driven autonomous AI agent stack backend and control dashboard."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

jarvis = Jarvis()

class CommandRequest(BaseModel):
    text: str

class CommandResponse(BaseModel):
    intent: str
    confidence: float
    response: str

@app.post("/command", response_model=CommandResponse)
async def handle_command_endpoint(req: CommandRequest):
    routed = route(req.text)
    response_text = await jarvis.handle_command(req.text)
    return CommandResponse(
        intent=routed["intent"],
        confidence=routed["confidence"],
        response=response_text
    )

@app.get("/health")
async def health_check():
    return {
        "status": "online",
        "service": "JARVIS AI Autonomous Stack",
        "version": "1.0.0",
        "whisper_model": settings.WHISPER_MODEL,
        "database": settings.DB_PATH,
    }

@app.get("/memory/recent")
async def recent_memory(limit: int = 10):
    return get_recent_conversations(limit)

@app.get("/memory/search")
async def search_memory_endpoint(q: str = ""):
    return search_memory(q)

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>JARVIS AI — Autonomous Desktop Agent Control Center</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #080b14;
            --card-bg: rgba(16, 22, 40, 0.65);
            --card-border: rgba(0, 240, 255, 0.15);
            --accent-cyan: #00f0ff;
            --accent-purple: #7000ff;
            --accent-pink: #ff007f;
            --text-main: #f0f4f8;
            --text-muted: #8a99ad;
            --glow-cyan: rgba(0, 240, 255, 0.35);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-dark);
            background-image: 
                radial-gradient(circle at 15% 15%, rgba(112, 0, 255, 0.15) 0%, transparent 45%),
                radial-gradient(circle at 85% 85%, rgba(0, 240, 255, 0.12) 0%, transparent 45%),
                radial-gradient(circle at 50% 50%, rgba(8, 11, 20, 0.95) 0%, #080b14 100%);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            overflow-x: hidden;
        }

        header {
            padding: 1.5rem 3rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(16px);
            background: rgba(8, 11, 20, 0.8);
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .logo-container {
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .logo-orb {
            width: 38px;
            height: 38px;
            border-radius: 50%;
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
            box-shadow: 0 0 20px var(--glow-cyan);
            animation: pulse-glow 3s infinite alternate;
        }

        @keyframes pulse-glow {
            0% { transform: scale(0.98); box-shadow: 0 0 15px rgba(0, 240, 255, 0.4); }
            100% { transform: scale(1.05); box-shadow: 0 0 28px rgba(112, 0, 255, 0.7); }
        }

        .logo-title {
            font-size: 1.5rem;
            font-weight: 800;
            letter-spacing: 2px;
            background: linear-gradient(90deg, #ffffff, var(--accent-cyan));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .status-badge {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            background: rgba(0, 240, 255, 0.08);
            border: 1px solid rgba(0, 240, 255, 0.2);
            padding: 0.5rem 1.2rem;
            border-radius: 30px;
            font-size: 0.85rem;
            font-weight: 600;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: #00ff88;
            box-shadow: 0 0 10px #00ff88;
            animation: blink 2s infinite;
        }

        @keyframes blink {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.3; }
        }

        main {
            flex: 1;
            padding: 2.5rem 3rem;
            max-width: 1400px;
            margin: 0 auto;
            width: 100%;
            display: grid;
            grid-template-columns: 1fr 380px;
            gap: 2rem;
        }

        .hero-section {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 24px;
            padding: 2.5rem;
            backdrop-filter: blur(20px);
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4);
            display: flex;
            flex-direction: column;
            gap: 2rem;
        }

        .hero-header h2 {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }

        .hero-header p {
            color: var(--text-muted);
            font-size: 1rem;
            line-height: 1.6;
        }

        .voice-control-area {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 3rem 1.5rem;
            background: rgba(0, 0, 0, 0.25);
            border: 1px dashed rgba(0, 240, 255, 0.2);
            border-radius: 20px;
            gap: 1.5rem;
            position: relative;
        }

        .mic-btn {
            width: 100px;
            height: 100px;
            border-radius: 50%;
            border: none;
            background: linear-gradient(135deg, var(--accent-purple), var(--accent-cyan));
            color: white;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 10px 30px rgba(0, 240, 255, 0.3);
            transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            position: relative;
        }

        .mic-btn:hover {
            transform: scale(1.08);
            box-shadow: 0 15px 40px rgba(112, 0, 255, 0.5);
        }

        .mic-btn svg {
            width: 42px;
            height: 42px;
        }

        .mic-waves {
            display: flex;
            align-items: center;
            gap: 4px;
            height: 30px;
        }

        .wave-bar {
            width: 4px;
            height: 10px;
            background: var(--accent-cyan);
            border-radius: 2px;
            transition: height 0.2s ease;
        }

        .mic-waves.active .wave-bar {
            animation: sound-wave 1s infinite ease-in-out alternate;
        }

        .mic-waves.active .wave-bar:nth-child(1) { animation-delay: 0.1s; }
        .mic-waves.active .wave-bar:nth-child(2) { animation-delay: 0.3s; }
        .mic-waves.active .wave-bar:nth-child(3) { animation-delay: 0.2s; }
        .mic-waves.active .wave-bar:nth-child(4) { animation-delay: 0.4s; }
        .mic-waves.active .wave-bar:nth-child(5) { animation-delay: 0.15s; }

        @keyframes sound-wave {
            0% { height: 6px; }
            100% { height: 28px; }
        }

        .input-bar-container {
            display: flex;
            gap: 1rem;
            width: 100%;
        }

        .cmd-input {
            flex: 1;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 14px;
            padding: 1rem 1.5rem;
            color: white;
            font-family: inherit;
            font-size: 1rem;
            outline: none;
            transition: all 0.3s ease;
        }

        .cmd-input:focus {
            border-color: var(--accent-cyan);
            box-shadow: 0 0 15px rgba(0, 240, 255, 0.25);
            background: rgba(255, 255, 255, 0.08);
        }

        .send-btn {
            background: linear-gradient(135deg, var(--accent-cyan), #00a8ff);
            color: #080b14;
            font-weight: 700;
            border: none;
            border-radius: 14px;
            padding: 0 2rem;
            cursor: pointer;
            font-size: 1rem;
            transition: all 0.2s ease;
        }

        .send-btn:hover {
            opacity: 0.9;
            transform: translateY(-2px);
            box-shadow: 0 5px 20px var(--glow-cyan);
        }

        .quick-chips {
            display: flex;
            flex-wrap: wrap;
            gap: 0.6rem;
        }

        .chip {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-size: 0.85rem;
            color: var(--text-muted);
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .chip:hover {
            border-color: var(--accent-cyan);
            color: var(--accent-cyan);
            background: rgba(0, 240, 255, 0.08);
        }

        .response-display {
            background: rgba(0, 0, 0, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 1.5rem;
            display: flex;
            flex-direction: column;
            gap: 0.8rem;
        }

        .response-meta {
            display: flex;
            justify-content: space-between;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--accent-cyan);
            font-family: 'JetBrains Mono', monospace;
        }

        .response-text {
            font-size: 1.1rem;
            line-height: 1.6;
            color: var(--text-main);
        }

        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 2rem;
        }

        .side-card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 24px;
            padding: 1.8rem;
            backdrop-filter: blur(20px);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        }

        .side-card h3 {
            font-size: 1.2rem;
            font-weight: 700;
            margin-bottom: 1rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .memory-list {
            display: flex;
            flex-direction: column;
            gap: 0.8rem;
            max-height: 380px;
            overflow-y: auto;
        }

        .memory-item {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 12px;
            padding: 0.8rem 1rem;
            font-size: 0.85rem;

        }

        .memory-item .intent-tag {
            font-size: 0.75rem;
            color: var(--accent-cyan);
            font-family: 'JetBrains Mono', monospace;
            margin-bottom: 0.3rem;
        }

        .memory-item .input-text {
            color: var(--text-main);
            font-weight: 500;
        }

        .docs-link {
            display: inline-block;
            margin-top: 1rem;
            color: var(--accent-cyan);
            text-decoration: none;
            font-weight: 600;
            font-size: 0.9rem;
            transition: underline 0.2s;
        }

        .docs-link:hover {
            text-decoration: underline;
        }

        @media (max-width: 1024px) {
            main {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <header>
        <div class="logo-container">
            <div class="logo-orb"></div>
            <div class="logo-title">JARVIS AI</div>
        </div>
        <div class="status-badge">
            <div class="status-dot"></div>
            <span>SYSTEM ACTIVE</span>
        </div>
    </header>

    <main>
        <section class="hero-section">
            <div class="hero-header">
                <h2>Autonomous Desktop AI Agent Stack</h2>
                <p>Voice-driven execution pipeline backed by Python, Whisper STT, ElevenLabs TTS, and SQLite Persistent Memory.</p>
            </div>

            <div class="voice-control-area">
                <button class="mic-btn" id="micBtn" onclick="toggleWebVoiceRecording()" title="Click to speak command">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z"/>
                        <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                        <line x1="12" y1="19" x2="12" y2="22"/>
                    </svg>
                </button>
                <div class="mic-waves" id="micWaves">
                    <div class="wave-bar"></div>
                    <div class="wave-bar"></div>
                    <div class="wave-bar"></div>
                    <div class="wave-bar"></div>
                    <div class="wave-bar"></div>
                </div>
                <p id="voiceStatus" style="font-size: 0.9rem; color: var(--text-muted);">Tap microphone to speak or type command below</p>
            </div>

            <div class="input-bar-container">
                <input type="text" class="cmd-input" id="cmdInput" placeholder="Ask JARVIS (e.g. 'What is the weather in Chennai?', 'Play focus music', 'Calculate 42 * 8')" onkeydown="if(event.key==='Enter') sendCommand()">
                <button class="send-btn" onclick="sendCommand()">EXECUTE</button>
            </div>

            <div class="quick-chips">
                <div class="chip" onclick="executeChip('What is the weather in Chennai?')">Weather in Chennai</div>
                <div class="chip" onclick="executeChip('Get latest top news headlines')">Top News</div>
                <div class="chip" onclick="executeChip('Play focus deep work music')">Focus Music</div>
                <div class="chip" onclick="executeChip('Calculate 250 * 45')">Math Calc</div>
                <div class="chip" onclick="executeChip('Open notepad')">Open Notepad</div>
                <div class="chip" onclick="executeChip('What did I ask recently?')">Recall Memory</div>
            </div>

            <div class="response-display">
                <div class="response-meta">
                    <span id="intentMeta">INTENT: STANDBY</span>
                    <span id="confMeta">CONFIDENCE: --</span>
                </div>
                <div class="response-text" id="responseText">
                    System ready. Awaiting natural voice input or text command...
                </div>
            </div>
        </section>

        <aside class="sidebar">
            <div class="side-card">
                <h3>
                    <span>Memory Recall</span>
                    <small style="color:var(--accent-cyan); cursor:pointer;" onclick="loadMemory()">Refresh</small>
                </h3>
                <div class="memory-list" id="memoryList">
                    <div class="memory-item">
                        <div class="intent-tag">LOADING MEMORY...</div>
                    </div>
                </div>
            </div>

            <div class="side-card">
                <h3>System Architecture</h3>
                <p style="font-size: 0.88rem; color: var(--text-muted); line-height: 1.5;">
                    • <b>STT:</b> OpenAI Whisper<br>
                    • <b>TTS:</b> ElevenLabs Monolingual v1<br>
                    • <b>Engine:</b> Python AsyncIO & Router<br>
                    • <b>DB:</b> SQLite3 persistent tables<br>
                    • <b>API:</b> FastAPI HTTP Server
                </p>
                <a href="/docs" target="_blank" class="docs-link">Explore Swagger API Docs &rarr;</a>
            </div>
        </aside>
    </main>

    <script>
        async function sendCommand(textOverride) {
            const input = document.getElementById('cmdInput');
            const query = textOverride || input.value.trim();
            if (!query) return;

            document.getElementById('intentMeta').innerText = 'INTENT: ROUTING...';
            document.getElementById('confMeta').innerText = 'PROCESSING';
            document.getElementById('responseText').innerText = 'Dispatching to module handler...';

            try {
                const res = await fetch('/command', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: query })
                });
                const data = await res.json();

                document.getElementById('intentMeta').innerText = `INTENT: ${data.intent.toUpperCase()}`;
                document.getElementById('confMeta').innerText = `CONFIDENCE: ${(data.confidence * 100).toFixed(0)}%`;
                document.getElementById('responseText').innerText = data.response;

                if (!textOverride) input.value = '';
                loadMemory();
            } catch (err) {
                document.getElementById('responseText').innerText = `Error executing command: ${err}`;
            }
        }

        function executeChip(text) {
            document.getElementById('cmdInput').value = text;
            sendCommand(text);
        }

        async function loadMemory() {
            try {
                const res = await fetch('/memory/recent?limit=6');
                const data = await res.json();
                const list = document.getElementById('memoryList');
                if (!data || data.length === 0) {
                    list.innerHTML = '<div class="memory-item"><div class="input-text">No conversation history yet.</div></div>';
                    return;
                }
                list.innerHTML = data.map(item => `
                    <div class="memory-item">
                        <div class="intent-tag">[${item.intent.toUpperCase()}] ${item.timestamp || ''}</div>
                        <div class="input-text">"${item.user_input}"</div>
                    </div>
                `).join('');
            } catch (e) {
                console.error("Failed to load memory:", e);
            }
        }

        // Web Speech Recognition for in-browser speech testing
        let recognition;
        function toggleWebVoiceRecording() {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            if (!SpeechRecognition) {
                alert("Browser Speech Recognition not supported in this browser. Please use text input or JARVIS local listener!");
                return;
            }
            if (!recognition) {
                recognition = new SpeechRecognition();
                recognition.continuous = false;
                recognition.interimResults = false;

                recognition.onstart = () => {
                    document.getElementById('micWaves').classList.add('active');
                    document.getElementById('voiceStatus').innerText = "Listening for voice command...";
                };

                recognition.onresult = (event) => {
                    const transcript = event.results[0][0].transcript;
                    document.getElementById('cmdInput').value = transcript;
                    sendCommand(transcript);
                };

                recognition.onerror = (e) => {
                    document.getElementById('voiceStatus').innerText = "Voice error: " + e.error;
                    document.getElementById('micWaves').classList.remove('active');
                };

                recognition.onend = () => {
                    document.getElementById('micWaves').classList.remove('active');
                    document.getElementById('voiceStatus').innerText = "Tap microphone to speak or type command below";
                };
            }
            recognition.start();
        }

        window.onload = loadMemory;
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.server:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True
    )
