import os
import tempfile
import numpy as np
from scipy.io import wavfile
from parakeet_mlx import from_pretrained
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn
import contextlib
import re
import asyncio
import uuid

# --- Configuration ---
MODEL_PATH = "mlx-community/parakeet-tdt-0.6b-v3"
SAMPLE_RATE = 16000
INFERENCE_INTERVAL = 1.0  # 💡 Throttle inference to execute exactly once per second

# Global shared resources
model = None
gpu_lock = asyncio.Lock()


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles application startup and shutdown lifecycle events safely."""
    global model
    print("Loading MLX Parakeet model (Apple Silicon GPU Accelerated)...")
    model = from_pretrained(MODEL_PATH)
    print("Model loaded successfully! Ready for multi-user connections.")
    yield

app = FastAPI(lifespan=lifespan)


def parse_transcript_result(result) -> str:
    """Parses various parakeet-mlx output types using structural pattern matching."""
    match result:
        case {"text": str(text)}: return text
        case {"tokens": list(tokens)}: return "".join([t.text for t in tokens if hasattr(t, 'text')])
        case [dict() as first_item, *_] if "text" in first_item: return first_item.get("text", "")
        case _ if hasattr(result, "text"): return getattr(result, "text", "")
        case _:
            raw_str = str(result)
            if "AlignedToken" in raw_str:
                matches = re.findall(r"text='([^']*)'", raw_str)
                return "".join(matches) if matches else re.sub(r"AlignedToken\(.*?\)", "", raw_str)
            return raw_str


async def inference_loop(websocket: WebSocket, session_id: str, user_audio_path: str, context_dict: dict):
    """
    💡 Dedicated background loop for throttled inference.
    Separates network data collection from heavy GPU computation.
    """
    while context_dict["active"]:
        await asyncio.sleep(INFERENCE_INTERVAL)

        # Skip if no new binary audio packets have arrived
        if not context_dict["has_new_data"] or len(context_dict["buffer"]) == 0:
            continue

        context_dict["has_new_data"] = False

        # Safely serialize GPU access using the global async lock
        async with gpu_lock:
            try:
                # Copy current buffer state snippet to avoid race conditions during disk IO write
                current_buffer = np.array(
                    context_dict["buffer"], dtype=np.float32)

                # Overwrite the isolated temporary wave block cache
                wavfile.write(user_audio_path, SAMPLE_RATE, current_buffer)

                # Execute hardware-accelerated MLX transcription sequentially
                result = model.transcribe(user_audio_path)
                clean_text = parse_transcript_result(
                    result).strip().replace(" ", " ")

                if clean_text and context_dict["active"]:
                    await websocket.send_json({"text": clean_text})
            except Exception as e:
                if context_dict["active"]:
                    await websocket.send_json({"text": f"[Inference Error]: {str(e)}"})

# --- WebSocket Streaming Endpoint ---


@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    """Handles isolated real-time audio streaming sessions for individual clients."""
    await websocket.accept()

    session_id = uuid.uuid4().hex
    temp_dir = tempfile.gettempdir()
    user_audio_path = os.path.join(temp_dir, f"stt_user_{session_id}.wav")

    # Thread-safe dictionary tracking mutable user context states
    context = {
        "buffer": np.zeros(0, dtype=np.float32),
        "has_new_data": False,
        "active": True
    }

    print(f"[Session Started] User tied to session: {session_id}")

    # 💡 Fire and forget the async throttled background loop worker task
    inference_task = asyncio.create_task(
        inference_loop(websocket, session_id, user_audio_path, context)
    )

    try:
        while True:
            # Fast non-blocking path: Simply ingest raw binary bytes straight into memory
            data = await websocket.receive_bytes()
            chunk = np.frombuffer(data, dtype=np.float32)
            context["buffer"] = np.concatenate((context["buffer"], chunk))

            # Sliding window context scaling: Caps maximum trail length to trailing 30s
            max_buffer = SAMPLE_RATE * 30
            if len(context["buffer"]) > max_buffer:
                context["buffer"] = context["buffer"][-max_buffer:]

            context["has_new_data"] = True

    except WebSocketDisconnect:
        print(f"[Session Ended] User disconnected: {session_id}")
    finally:
        # Tear down background tasks and perform final garbage cleanup routines
        context["active"] = False
        inference_task.cancel()
        if os.path.exists(user_audio_path):
            with contextlib.suppress(FileNotFoundError):
                os.remove(user_audio_path)

# --- Multi-User Responsive Web UI ---


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Multi-User Local Transcriber</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                background-color: #121212;
                color: #e0e0e0;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                height: 100vh;
                margin: 0;
            }
            .container {
                width: 90%;
                max-width: 650px;
                background: #1e1e1e;
                padding: 30px;
                border-radius: 12px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.5);
                text-align: center;
            }
            h1 { font-size: 24px; margin-bottom: 5px; color: #fff; }
            p.subtitle { color: #888; font-size: 13px; margin-bottom: 25px; }
            .controls { display: flex; justify-content: center; align-items: center; gap: 15px; margin-bottom: 20px; }
            button {
                padding: 12px 24px;
                font-size: 15px;
                font-weight: bold;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                transition: background 0.2s;
            }
            .btn-start { background-color: #1f6aa5; color: white; }
            .btn-start:hover { background-color: #144870; }
            .btn-stop { background-color: #c0392b; color: white; display: none; }
            .btn-stop:hover { background-color: #a93226; }
            .status-tag { font-size: 14px; color: #888; }
            .status-tag.active { color: #2ecc71; font-weight: bold; }
            textarea {
                width: 100%;
                height: 260px;
                background-color: #151515;
                color: #ffffff;
                border: 1px solid #2d2d2d;
                border-radius: 8px;
                padding: 18px;
                font-size: 16px;
                line-height: 1.6;
                box-sizing: border-box;
                resize: none;
                font-family: inherit;
            }
            textarea:focus { outline: none; border-color: #444; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎙️ Multi-User Local Transcriber</h1>
            <p class="subtitle">Distributed Browser-Audio Capturing over Throttled WebSockets</p>
            
            <div class="controls">
                <button id="startBtn" class="btn-start" onclick="startSession()">Start Streaming</button>
                <button id="stopBtn" class="btn-stop" onclick="stopSession()">Stop</button>
                <span id="statusLabel" class="status-tag">Status: Idle</span>
            </div>

            <textarea id="outputBox" placeholder="Your clean local live transcription output will render here..." readonly></textarea>
        </div>

        <script>
            let ws = null;
            let audioContext = null;
            let processor = null;
            let globalStream = null;

            const outputBox = document.getElementById('outputBox');
            const startBtn = document.getElementById('startBtn');
            const stopBtn = document.getElementById('stopBtn');
            const statusLabel = document.getElementById('statusLabel');

            async function startSession() {
                outputBox.value = "";
                
                const wsProtocol = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
                ws = new WebSocket(`${wsProtocol}${window.location.host}/ws/stream`);
                
                ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    if (data.text) {
                        outputBox.value = data.text;
                        outputBox.scrollTop = outputBox.scrollHeight;
                    }
                };

                ws.onopen = async () => {
                    try {
                        globalStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
                        audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
                        const source = audioContext.createMediaStreamSource(globalStream);
                        processor = audioContext.createScriptProcessor(2048, 1, 1);
                        
                        processor.onaudioprocess = (e) => {
                            if (ws.readyState === WebSocket.OPEN) {
                                const inputData = e.inputBuffer.getChannelData(0);
                                ws.send(inputData.buffer);
                            }
                        };

                        source.connect(processor);
                        processor.connect(audioContext.destination);

                        startBtn.style.display = 'none';
                        stopBtn.style.display = 'block';
                        statusLabel.innerText = "🔴 Listening... Your local mic is streaming directly to host.";
                        statusLabel.className = "status-tag active";
                    } catch (err) {
                        alert("Microphone access blocked or failed: " + err);
                        ws.close();
                    }
                };

                ws.onclose = () => stopSession();
            }

            function stopSession() {
                if (processor) { processor.disconnect(); processor = null; }
                if (audioContext) { audioContext.close(); audioContext = null; }
                if (globalStream) { globalStream.getTracks().forEach(track => track.stop()); globalStream = null; }
                if (ws && ws.readyState === WebSocket.OPEN) { ws.close(); }
                
                startBtn.style.display = 'block';
                stopBtn.style.display = 'none';
                statusLabel.innerText = "Status: Idle";
                statusLabel.className = "status-tag";
            }
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
