import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from transformers import AutoTokenizer
from PIL import Image
import numpy as np
import cv2
import os

from cog import BasePredictor, Input, Path

class Predictor(BasePredictor):
    def setup(self):
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-scribble", torch_dtype=torch.float16
        )
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            safety_checker=None,
            torch_dtype=torch.float16,
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to("cuda")

    def predict(self, image: Path = Input(description="Upload an image")) -> Path:
        input_image = Image.open(image).convert("RGB")
        np_image = np.array(input_image)
        gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, threshold1=100, threshold2=200)
        canny_image = Image.fromarray(edges)

        result = self.pipe(
            prompt="black and white lineart drawing, high detail, clean lines, for adult coloring books",
            image=canny_image,
            num_inference_steps=20
        ).images[0]

        out_path = "/tmp/result.png"
        result.save(out_path)
        return Path(out_path)
