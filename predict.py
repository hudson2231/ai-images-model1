import os
import torch
import numpy as np
from cog import BasePredictor, Input, Path
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from PIL import Image
import requests
from io import BytesIO

class Predictor(BasePredictor):
    def setup(self):
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-hed",
            torch_dtype=torch.float16
        )

        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch.float16
        ).to("cuda")

        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_xformers_memory_efficient_attention()

    def predict(
        self,
        input_image: Path = Input(description="Photo to convert to coloring page line art"),
        prompt: str = Input(
            description="Prompt for generation",
            default="Highly detailed black and white line art of the entire photo. Preserve all facial features and background. Clean lines only, no shading, coloring book style."
        ),
        num_inference_steps: int = Input(description="Steps for generation", default=25),
        guidance_scale: float = Input(description="Prompt adherence", default=12.0),
        seed: int = Input(description="Random seed", default=42),
    ) -> Path:
        torch.manual_seed(seed)

        image = Image.open(input_image).convert("RGB")

        output = self.pipe(
            image=image,
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]

        output_path = "/tmp/output.png"
        output.save(output_path)
        return Path(output_path)
