"""Alpamayo model inference."""

import copy
from typing import Tuple, Optional

import torch
import numpy as np
import cv2

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper


class AlpamayoInference:
    """Alpamayo R1 model inference engine."""

    def __init__(self, model_path: str, device: str = "cuda", dtype: str = "bfloat16"):
        self.model_path = model_path
        self.device = device
        self.dtype = getattr(torch, dtype)
        self.model: Optional[AlpamayoR1] = None
        self.processor = None

    def load(self):
        """Load the model."""
        print(f"Loading Alpamayo model from {self.model_path}...")
        self.model = AlpamayoR1.from_pretrained(
            self.model_path, dtype=self.dtype
        ).to(self.device)
        self.processor = helper.get_processor(self.model.tokenizer)
        print("Model loaded!")

    def preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess frame for model input."""
        # Resize to expected size
        img = cv2.resize(frame, (1920, 1080))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Convert to tensor [C, H, W]
        tensor = torch.from_numpy(img).permute(2, 0, 1)

        # Stack 4 frames (temporal)
        tensor = tensor.unsqueeze(0).unsqueeze(0).repeat(1, 4, 1, 1, 1)

        return tensor

    def infer(
        self,
        frame: np.ndarray,
        temperature: float = 0.6,
        top_p: float = 0.98,
        max_length: int = 32
    ) -> Tuple[np.ndarray, str]:
        """
        Run inference on a frame.

        Returns:
            trajectory: [N, 3] array of predicted positions
            coc: Chain-of-causation text
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # Preprocess
        img_tensor = self.preprocess(frame)

        # Create model input
        messages = helper.create_message(img_tensor.flatten(0, 1))
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False,
            continue_final_message=True, return_dict=True, return_tensors="pt"
        )

        model_inputs = {
            "tokenized_data": copy.deepcopy(dict(inputs)),
            "ego_history_xyz": torch.zeros(1, 1, 16, 3),
            "ego_history_rot": torch.eye(3).unsqueeze(0).repeat(16, 1, 1).unsqueeze(0).unsqueeze(0),
        }
        model_inputs = helper.to_device(model_inputs, self.device)

        # Run inference
        with torch.no_grad(), torch.autocast("cuda", dtype=self.dtype):
            pred_xyz, pred_rot, extra = self.model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs,
                top_p=top_p,
                temperature=temperature,
                num_traj_samples=1,
                max_generation_length=max_length,
                return_extra=True,
            )

        trajectory = pred_xyz[0, 0, 0].cpu().numpy()
        coc = extra["cot"][0][0][0] if extra["cot"][0][0][0] else ""

        return trajectory, coc
