# Speech-to-Text (STT) Application

A real-time speech-to-text application powered by **Parakeet MLX** with GPU acceleration on Apple Silicon devices. Built with **FastAPI** and **WebSockets** for seamless real-time transcription.

## Features

✨ **Real-time Transcription** - Stream audio and get live transcription results
🚀 **GPU Accelerated** - Apple Silicon (MLX) GPU optimization for fast inference
🔄 **WebSocket Support** - Continuous connection for real-time processing
🎙️ **Multi-user** - Handle multiple simultaneous connections
⚡ **Throttled Inference** - Efficient inference execution (1 transcription per second)
🎯 **High Accuracy** - Parakeet TDT 0.6B v3 model for accurate transcription

## Requirements

- Python 3.13+
- Apple Silicon Mac (MLX GPU acceleration)
- Audio input device

## Installation

### Prerequisites

- Install `uv` if you haven't already:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

1. **Clone or navigate to the project directory**

   ```bash
   cd local-speech-to-text
   ```

2. **Create and activate a virtual environment**

   ```bash
   uv venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   uv sync
   ```

## Usage

1. **Start the server**

   ```bash
   uv run app.py
   ```

   The application will:
   - Load the Parakeet MLX model
   - Start the FastAPI server (typically at `http://localhost:8000`)
   - Be ready for WebSocket connections

2. **Connect via WebSocket**
   - Send audio data through a WebSocket connection
   - Receive real-time transcription results

## Technical Stack

| Component               | Technology                 |
| ----------------------- | -------------------------- |
| Web Framework           | FastAPI                    |
| Real-time Communication | WebSockets                 |
| Speech-to-Text Model    | Parakeet MLX (TDT 0.6B v3) |
| Audio Processing        | SciPy                      |
| GPU Acceleration        | MLX (Apple Silicon)        |
| Server                  | Uvicorn                    |

## Project Structure

```
local-speech-to-text/
├── app.py               # FastAPI application with WebSocket endpoint
├── pyproject.toml       # Project configuration and dependencies
└── README.md            # This file
```

## API Details

### WebSocket Endpoint

- **Path**: `/ws/{client_id}`
- **Protocol**: WebSocket
- **Message Format**: Audio data (WAV format)
- **Response**: Transcription text

## Performance Notes

- Model: Parakeet TDT 0.6B v3
- Sample Rate: 16000 Hz
- Inference Throttling: 1 inference per second
- GPU: MLX-accelerated on Apple Silicon

## Dependencies

- **fastapi** - Modern web framework for building APIs
- **parakeet-mlx** - Speech-to-text model with MLX acceleration
- **scipy** - Scientific computing and audio I/O
- **sounddevice** - Audio recording capability
- **uvicorn** - ASGI web server
- **websockets** - WebSocket implementation
