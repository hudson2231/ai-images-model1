import os
import cv2
import torch
import numpy as np
from cog import BasePredictor, Input, Path
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from transformers import AutoTokenizer, AutoFeatureExtractor, CLIPImageProcessor
from PIL import Image

class Predictor(BasePredictor):
    def setup(self):
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-scribble",
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
        input_image: Path = Input(description="Uploaded photo to convert to line art"),
        prompt: str = Input(description="Prompt for style", default="black and white line art for adult coloring book"),
        num_inference_steps: int = Input(description="Steps for generation", default=20),
        guidance_scale: float = Input(description="Classifier-free guidance scale", default=9.0),
        seed: int = Input(description="Random seed (for reproducibility)", default=42),
    ) -> Path:
        torch.manual_seed(seed)

        # Load and process input image
        image = Image.open(input_image).convert("RGB")
        np_image = np.array(image)

        # Generate edge map using Canny
        low_threshold, high_threshold = 100, 200
        edges = cv2.Canny(np_image, low_threshold, high_threshold)
        edge_rgb = np.stack([edges]*3, axis=-1)
        edge_pil = Image.fromarray(edge_rgb)

        # Run the model
        output = self.pipe(
            prompt,
            image=edge_pil,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        ).images[0]

        output_path = "/tmp/output.png"
        output.save(output_path)
        return Path(output_path)
