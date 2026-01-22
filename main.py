#!/usr/bin/env python3
"""
Alpamayo Robot Server - Tesla FSD Style Interface.

A FastAPI-based web server for controlling robots with
Alpamayo R1 autonomous driving model.
"""

import asyncio
import time
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse

import config
from robot.ros_client import ROSClient
from alpamayo.inference import AlpamayoInference
from visualization.tesla_viz import render_frame


# ============================================================
# Application State
# ============================================================
class AppState:
    """Global application state."""

    def __init__(self):
        self.ros_client: Optional[ROSClient] = None
        self.inference: Optional[AlpamayoInference] = None

        self.running = True
        self.ai_enabled = False

        # Latest visualization
        self.viz_frame: Optional[np.ndarray] = None
        self.viz_lock = threading.Lock()

        # Stats
        self.fps = 0.0
        self.coc_text = ""
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        self.trajectory: Optional[np.ndarray] = None


state = AppState()


# ============================================================
# FastAPI Application
# ============================================================
app = FastAPI(
    title="Alpamayo Robot Server",
    description="Tesla FSD-style autonomous robot control",
    version="1.0.0"
)

# Serve static files
static_path = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_path), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve main page."""
    html_path = static_path / "index.html"
    return html_path.read_text()


@app.get("/api/autopilot/toggle")
async def toggle_autopilot():
    """Toggle autopilot mode."""
    state.ai_enabled = not state.ai_enabled
    if not state.ai_enabled:
        state.ros_client.stop()
    return {"enabled": state.ai_enabled}


@app.get("/api/autopilot/status")
async def autopilot_status():
    """Get autopilot status."""
    return {
        "enabled": state.ai_enabled,
        "fps": state.fps,
        "coc": state.coc_text
    }


@app.get("/api/stop")
async def stop_robot():
    """Emergency stop."""
    state.ai_enabled = False
    if state.ros_client:
        state.ros_client.stop()
    return {"status": "stopped"}


@app.get("/api/move")
async def move_robot(
    linear: float = Query(0.0, ge=-1.0, le=1.0),
    angular: float = Query(0.0, ge=-2.0, le=2.0)
):
    """Manual movement control."""
    if not state.ai_enabled and state.ros_client:
        # Clamp values
        linear = max(-config.CONTROL.max_linear_vel,
                     min(config.CONTROL.max_linear_vel, linear))
        angular = max(-config.CONTROL.max_angular_vel,
                      min(config.CONTROL.max_angular_vel, angular))

        state.ros_client.send_velocity(linear, angular)
        state.linear_vel = linear
        state.angular_vel = angular

    return {"status": "ok", "linear": linear, "angular": angular}


def generate_video_stream():
    """Generate MJPEG video stream."""
    while state.running:
        with state.viz_lock:
            frame = state.viz_frame

        if frame is None:
            # Create placeholder
            frame = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Connecting...", (220, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (80, 80, 80), 2)

        # Encode to JPEG
        _, buffer = cv2.imencode('.jpg', frame,
                                  [cv2.IMWRITE_JPEG_QUALITY, config.SERVER.video_quality])

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        time.sleep(1.0 / config.SERVER.video_fps)


@app.get("/video")
async def video_feed():
    """MJPEG video stream endpoint."""
    return StreamingResponse(
        generate_video_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# ============================================================
# Inference Thread
# ============================================================
def trajectory_to_velocity(traj: np.ndarray) -> tuple:
    """Convert trajectory to velocity command."""
    if traj is None or len(traj) < 2:
        return 0.0, 0.0

    lookahead = min(config.CONTROL.lookahead_steps, len(traj) - 1)
    target = traj[lookahead]

    forward = target[0]
    lateral = target[1]

    linear = np.clip(
        forward * config.CONTROL.linear_scale,
        -config.CONTROL.max_linear_vel,
        config.CONTROL.max_linear_vel
    )

    if forward > 0.05:
        angular = np.arctan2(lateral, forward) * config.CONTROL.angular_scale
    else:
        angular = 0.0

    angular = np.clip(angular, -config.CONTROL.max_angular_vel, config.CONTROL.max_angular_vel)

    return float(linear), float(angular)


def inference_loop():
    """Main inference loop running in background thread."""
    print("Inference thread started")

    while state.running:
        # Get frame from robot
        frame = state.ros_client.get_frame() if state.ros_client else None

        if frame is None:
            time.sleep(0.1)
            continue

        try:
            # Run inference
            t0 = time.time()
            traj, coc = state.inference.infer(
                frame,
                temperature=config.MODEL.temperature,
                top_p=config.MODEL.top_p,
                max_length=config.MODEL.max_generation_length
            )
            inference_time = time.time() - t0

            # Update state
            state.trajectory = traj
            state.coc_text = coc
            state.fps = 1.0 / inference_time if inference_time > 0 else 0

            # Control robot if AI enabled
            if state.ai_enabled:
                linear, angular = trajectory_to_velocity(traj)
                state.ros_client.send_velocity(linear, angular)
                state.linear_vel = linear
                state.angular_vel = angular

            # Render visualization
            viz = render_frame(
                frame,
                traj,
                state.fps,
                coc,
                state.linear_vel,
                state.angular_vel,
                state.ai_enabled
            )

            with state.viz_lock:
                state.viz_frame = viz

            # Log
            print(f"\r[{state.fps:.1f}fps] AI:{state.ai_enabled} "
                  f"v={state.linear_vel:+.2f} w={state.angular_vel:+.2f} | "
                  f"{coc[:40]}...", end="", flush=True)

        except Exception as e:
            print(f"\nInference error: {e}")
            import traceback
            traceback.print_exc()
            if state.ros_client:
                state.ros_client.stop()
            time.sleep(0.5)


# ============================================================
# Startup & Shutdown
# ============================================================
@app.on_event("startup")
async def startup():
    """Initialize on server startup."""
    print("=" * 60)
    print("     ALPAMAYO R1 - Tesla FSD Style Interface")
    print("=" * 60)

    # Initialize ROS client
    print(f"\nConnecting to ROS at {config.ROBOT.ros_host}:{config.ROBOT.ros_port}...")
    state.ros_client = ROSClient(config.ROBOT.ros_host, config.ROBOT.ros_port)

    if state.ros_client.connect():
        state.ros_client.setup_publishers(config.ROBOT.cmd_vel_topic)
        state.ros_client.subscribe_camera(config.ROBOT.camera_topic)
        print("ROS connected!")
    else:
        print("Warning: ROS connection failed")

    # Wait for camera
    print("Waiting for camera...")
    for _ in range(50):
        if state.ros_client.get_frame() is not None:
            break
        await asyncio.sleep(0.1)

    if state.ros_client.get_frame() is None:
        print("Warning: No camera feed")
    else:
        print("Camera OK!")

    # Load model
    state.inference = AlpamayoInference(
        config.MODEL.model_path,
        config.MODEL.device,
        config.MODEL.dtype
    )
    state.inference.load()

    # Start inference thread
    inf_thread = threading.Thread(target=inference_loop, daemon=True)
    inf_thread.start()

    print("\n" + "=" * 60)
    print(f"   Server running at http://{config.SERVER.host}:{config.SERVER.port}")
    print("=" * 60 + "\n")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on server shutdown."""
    print("\nShutting down...")
    state.running = False

    if state.ros_client:
        state.ros_client.stop()
        state.ros_client.disconnect()

    print("Goodbye!")


# ============================================================
# Main Entry Point
# ============================================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=config.SERVER.host,
        port=config.SERVER.port,
        reload=False,
        log_level="info"
    )
