import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

from model import SimpleTokenizer, TylerVLAPolicy

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
_robot: SO101Follower | None = None


def load_policy(run_dir: str, device: str = None):
    run_dir = Path(run_dir)
    ckpt = torch.load(run_dir / "model.pt", map_location="cpu")
    with open(run_dir / "tokenizer.json", "r") as f:
        vocab = json.load(f)["vocab"]
    tokenizer = SimpleTokenizer(vocab)
    norm = np.load(run_dir / "joint_norm.npz")
    j_mean = norm["mean"].astype(np.float32)
    j_std = norm["std"].astype(np.float32)

    device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = TylerVLAPolicy(vocab_size=ckpt["vocab_size"], num_joints=ckpt["num_joints"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, tokenizer, j_mean, j_std, device


def preprocess_image(img_hwc_uint8: np.ndarray, image_size: int = 128) -> torch.Tensor:
    # img: HWC uint8 -> torch CHW float [0,1]
    x = torch.from_numpy(img_hwc_uint8).permute(2, 0, 1).float() / 255.0
    x = F.interpolate(x.unsqueeze(0), size=(image_size, image_size),
                      mode="bilinear", align_corners=False).squeeze(0)
    return x


# ---- SO-ARM-101 via LeRobot ----
def init_robot(port: str = "/dev/tty.usbmodem5A460830061", camera_index: int = 0):
    global _robot
    config = SO101FollowerConfig(
        port=port,
        cameras={
            "front": OpenCVCameraConfig(
                index_or_path=camera_index, fps=5, width=1920, height=1080,
            )
        },
    )
    _robot = SO101Follower(config)
    _robot.connect(calibrate=False)


def disconnect_robot():
    global _robot
    if _robot is not None and _robot.is_connected:
        _robot.disconnect()
        _robot = None


def get_rgb_frame() -> np.ndarray:
    """Return an HWC uint8 RGB frame from the camera."""
    return _robot.cameras["front"].async_read()


def get_current_joint_positions() -> np.ndarray:
    """Return current joint positions [6] float32 in LeRobot normalized space."""
    pos = _robot.bus.sync_read("Present_Position")
    return np.array([pos[name] for name in JOINT_NAMES], dtype=np.float32)


def set_joint_positions(q_des: np.ndarray):
    """Send desired joint positions [6] to SO-ARM-101."""
    action = {f"{name}.pos": float(q_des[i]) for i, name in enumerate(JOINT_NAMES)}
    _robot.send_action(action)
# --------------------------------


def main(run_dir: str, command: str, hz: float = 10.0):
    model, tokenizer, j_mean, j_std, device = load_policy(run_dir)
    dt = 1.0 / hz

    init_robot()

    # Cache text tokens once (command usually constant per episode)
    text_ids = tokenizer.encode(command, max_len=16).unsqueeze(0).to(device)

    # Simple smoothing to reduce jitter
    alpha = 0.2  # 0..1; higher = more responsive, lower = smoother
    q_prev = None

    try:
        while True:
            t0 = time.time()

            img = get_rgb_frame()
            img_t = preprocess_image(img, image_size=128).unsqueeze(0).to(device)

            with torch.no_grad():
                pred_norm = model(img_t, text_ids).squeeze(0).cpu().numpy()  # [J] normalized
            q_des = pred_norm * j_std + j_mean  # de-normalize to LeRobot space

            # Safety / smoothing
            if q_prev is None:
                q_prev = get_current_joint_positions().astype(np.float32)
            q_cmd = (1 - alpha) * q_prev + alpha * q_des
            q_prev = q_cmd

            # Optional: clamp to joint limits if you have them
            # q_cmd = np.clip(q_cmd, q_min, q_max)

            set_joint_positions(q_cmd)

            # Rate control
            elapsed = time.time() - t0
            sleep_t = max(0.0, dt - elapsed)
            time.sleep(sleep_t)
    finally:
        disconnect_robot()


if __name__ == "__main__":
    main("runs/pick_place_v1", command="pick up the ball and place it in the bowl", hz=10.0)