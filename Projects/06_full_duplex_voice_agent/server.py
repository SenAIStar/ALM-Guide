import uuid
from fastapi import FastAPI, WebSocket
from state_machine import DuplexSession

app = FastAPI()

@app.websocket("/voice")
async def voice_socket(socket: WebSocket):
    await socket.accept()
    session = DuplexSession()
    while True:
        message = await socket.receive_json()
        event = message["event"]
        if event == "speech":
            action = session.on_user_audio(True)
            await socket.send_json({"event": action, "state": session.state})
        elif event == "endpoint":
            session.on_endpoint()
            generation_id = str(uuid.uuid4())
            session.on_generation_started(generation_id)
            await socket.send_json({"event": "generation_started", "generation_id": generation_id})
        elif event == "cancelled":
            session.on_cancelled(message["generation_id"])
            await socket.send_json({"event": "ready", "state": session.state})
