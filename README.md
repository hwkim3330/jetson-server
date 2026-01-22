# Alpamayo Robot Server

Tesla FSD-style autonomous robot control interface powered by NVIDIA Alpamayo R1 10B VLA model.

## Features

- **Real-time AI Inference**: Alpamayo R1 trajectory prediction at ~1 FPS
- **Tesla FSD Visualization**: Path overlay, HUD, steering indicator
- **Dual Control Modes**: AI autopilot or manual control
- **Web Interface**: Access from any browser, mobile-friendly
- **ROS Integration**: WebSocket-based communication

## Requirements

- NVIDIA GPU with CUDA support (RTX 3090+ recommended)
- Python 3.10+
- ROS robot with rosbridge WebSocket server
- Alpamayo R1 model weights

## Installation

```bash
# Clone repository
git clone https://github.com/hwkim3330/jetson-server.git
cd jetson-server

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install Alpamayo R1
pip install git+https://github.com/NVlabs/alpamayo-r1.git
```

## Configuration

Edit `config.py` to match your setup:

```python
# Robot connection
ROBOT.ros_host = "192.168.10.1"  # Robot IP
ROBOT.ros_port = 9090             # rosbridge port

# Model
MODEL.model_path = "/path/to/alpamayo-r1-10b"

# Server
SERVER.port = 8899
```

## Usage

```bash
# Activate environment
source venv/bin/activate

# Run server
python main.py
```

Open browser: `http://localhost:8899`

## Controls

### Web Interface
- **Autopilot ON/OFF**: Toggle AI autonomous driving
- **STOP**: Emergency stop
- **Arrow buttons**: Manual movement

### Keyboard
- `W/A/S/D` or Arrow keys: Drive
- `Space`: Emergency stop

## Architecture

```
jetson-server/
├── main.py              # FastAPI server entry point
├── config.py            # Configuration settings
├── requirements.txt
├── alpamayo/
│   └── inference.py     # Alpamayo model wrapper
├── robot/
│   └── ros_client.py    # ROS WebSocket client
├── visualization/
│   └── tesla_viz.py     # Tesla FSD-style rendering
└── static/
    └── index.html       # Web UI
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/video` | GET | MJPEG video stream |
| `/api/autopilot/toggle` | GET | Toggle AI mode |
| `/api/autopilot/status` | GET | Get AI status |
| `/api/stop` | GET | Emergency stop |
| `/api/move` | GET | Manual control (`?linear=0.3&angular=0.5`) |

## Performance Benchmark

Tested on RTX 3090 (24GB VRAM), Ubuntu 22.04, CUDA 12.8, PyTorch 2.10.0

### Inference Time (Alpamayo R1 10B)

| Configuration | Time | FPS | Notes |
|---------------|------|-----|-------|
| 1 cam, 1 frame | 1.09s | 0.9 | Minimal input |
| 1 cam, 4 frames | 0.75s | 1.3 | Recommended |
| 4 cam, 4 frames | 2.20s | 0.5 | Full sensor suite |
| Web server (real-time) | 1.4-1.6s | 0.6-0.7 | With visualization overhead |

### Bottleneck Analysis

| Component | Time | % of Total |
|-----------|------|------------|
| VLM Text Generation | ~1.3s | 85-90% |
| FlowMatching Diffusion | ~0.15s | 10-15% |
| Preprocessing | ~0.05s | <5% |

### Official NVIDIA Specs

- Target latency: **99ms** (on DRIVE Orin/Thor)
- DRIVE Orin: 254 TOPS @ 65W
- DRIVE Thor: 1000+ TOPS @ 100W

### Optimization Notes

Current RTX 3090 performance is ~10x slower than target due to:
1. VLM autoregressive generation bottleneck
2. No TensorRT optimization for VLM
3. BF16 inference (not INT8/FP8 quantized)

For real-time autonomous driving, requires:
- TensorRT-LLM optimization
- INT8/FP8 quantization
- NVIDIA DRIVE platform

## License

Apache 2.0

## Credits

- [NVIDIA Alpamayo R1](https://github.com/NVlabs/alpamayo-r1)
- [rosbridge_suite](http://wiki.ros.org/rosbridge_suite)
