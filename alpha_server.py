import os
from typing import Dict, List
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel
from groq import Groq

load_dotenv()

app = FastAPI(title="A.L.P.H.A. // Neural & P2P Core", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, "web")
os.makedirs(WEB_DIR, exist_ok=True)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

MODELS = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b"]


class AskRequest(BaseModel):
    prompt: str


class RoomManager:
    def __init__(self):
        self.rooms: Dict[str, List[WebSocket]] = {}

    async def join(self, room_id: str, ws: WebSocket):
        await ws.accept()
        if room_id not in self.rooms:
            self.rooms[room_id] = []
        self.rooms[room_id].append(ws)
        if len(self.rooms[room_id]) > 1:
            for peer in self.rooms[room_id]:
                if peer != ws:
                    await peer.send_json({"type": "peer_joined"})

    def leave(self, room_id: str, ws: WebSocket):
        if room_id in self.rooms and ws in self.rooms[room_id]:
            self.rooms[room_id].remove(ws)
            if not self.rooms[room_id]:
                del self.rooms[room_id]

    async def relay(self, room_id: str, sender: WebSocket, payload: dict):
        for peer in list(self.rooms.get(room_id, [])):
            if peer != sender:
                try:
                    await peer.send_json(payload)
                except Exception:
                    pass


room_mgr = RoomManager()


@app.get("/", response_class=HTMLResponse)
async def home():
    idx = os.path.join(WEB_DIR, "index.html")
    if not os.path.exists(idx):
        return HTMLResponse("<h3>A.L.P.H.A. Core Online</h3>")
    with open(idx, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/alpha_logo.svg")
async def alpha_logo():
    svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">
      <rect width="256" height="256" rx="56" fill="#02060d"/>
      <circle cx="128" cy="128" r="96" fill="none" stroke="#00f0ff" stroke-width="8" stroke-dasharray="28 12"/>
      <circle cx="128" cy="128" r="68" fill="none" stroke="#ff1e7a" stroke-width="6" stroke-dasharray="16 14"/>
      <circle cx="128" cy="128" r="34" fill="#00f0ff" fill-opacity="0.25" stroke="#00f0ff" stroke-width="5"/>
      <text x="128" y="139" font-family="Consolas, monospace" font-size="34" font-weight="bold" fill="#ffffff" text-anchor="middle">A</text>
    </svg>"""
    return Response(content=svg, media_type="image/svg+xml")


@app.get("/manifest.json")
async def manifest():
    return JSONResponse({
        "name": "QuantumLab Physics HUD",
        "short_name": "QuantumLab",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#02060d",
        "theme_color": "#00f0ff",
        "icons": [
            {
                "src": "/alpha_logo.svg",
                "sizes": "any",
                "type": "image/svg+xml",
                "purpose": "any maskable"
            }
        ]
    })


@app.get("/sw.js")
async def service_worker():
    js = "self.addEventListener('fetch', function(event) {});"
    return Response(content=js, media_type="application/javascript")


@app.get("/ping")
async def ping():
    return {"status": "alive", "core": "ALPHA"}


@app.post("/api/ask")
async def ask_alpha(req: AskRequest):
    if not client:
        return {"reply": "GROQ_API_KEY missing in server environment."}
    for m in MODELS:
        try:
            res = client.chat.completions.create(
                model=m,
                messages=[
                    {"role": "system", "content": "You are A.L.P.H.A., a sharp, concise AI academic and coding assistant built by Aris Industries."},
                    {"role": "user", "content": req.prompt}
                ],
                temperature=0.6,
                max_tokens=1000,
            )
            return {"reply": res.choices[0].message.content.strip()}
        except Exception:
            continue
    return {"reply": "Neural core busy, try again in a second."}


@app.websocket("/ws/call/{room_id}")
async def call_signaling(ws: WebSocket, room_id: str):
    rid = room_id.strip().lower()
    await room_mgr.join(rid, ws)
    try:
        while True:
            data = await ws.receive_json()
            await room_mgr.relay(rid, ws, data)
    except WebSocketDisconnect:
        room_mgr.leave(rid, ws)
        await room_mgr.relay(rid, ws, {"type": "peer_left"})
    except Exception:
        room_mgr.leave(rid, ws)
        await room_mgr.relay(rid, ws, {"type": "peer_left"})
