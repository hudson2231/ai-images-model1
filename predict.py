import os
import cv2
import torch
import numpy as np
from cog import BasePredictor, Input, Path
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from PIL import Image
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation

class Predictor(BasePredictor):
    def setup(self):
        # Load ControlNet HED model
        self.controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-hed",
            torch_dtype=torch.float16
        )

        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=self.controlnet,
            torch_dtype=torch.float16
        ).to("cuda")

        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_xformers_memory_efficient_attention()

    def predict(
        self,
        input_image: Path = Input(description="Uploaded photo to convert to line art"),
        prompt: str = Input(
            description="Prompt for style",
            default="Detailed clean black and white line art drawing of the subject and background. No shading. High detail. Coloring book page style."
        ),
        num_inference_steps: int = Input(description="Steps for generation", default=25),
        guidance_scale: float = Input(description="Classifier-free guidance scale", default=12.5),
        seed: int = Input(description="Random seed (for reproducibility)", default=42),
    ) -> Path:
        torch.manual_seed(seed)

        # Load and process input image
        image = Image.open(input_image).convert("RGB")
        np_image = np.array(image)

        # --- HED Detection (cleaner than canny) ---
        import cv2
        from diffusers.utils import load_image
        from diffusers.utils import load_image, make_image_grid
        from transformers import pipeline as transformers_pipeline

        # Use the Diffusers' builtin HED processor
        from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
        from diffusers.utils import load_image
        from diffusers.pipelines.controlnet import StableDiffusionControlNetPipeline
        from diffusers.pipelines.controlnet import ControlNetModel, ControlNetConditioningImageTransform

        from controlnet_aux import HEDdetector

        hed = HEDdetector.from_pretrained("lllyasviel/ControlNet")

        edge_image = hed(image)
        edge_pil = edge_image.resize(image.size)

        # Run generation
        output = self.pipe(
            prompt=prompt,
            image=edge_pil,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        ).images[0]

        output_path = "/tmp/output.png"
        output.save(output_path)
        return Path(output_path)

