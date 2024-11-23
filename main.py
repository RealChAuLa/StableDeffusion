from diffusers import StableDiffusionPipeline
import torch

# Load the model
model_id = "stabilityai/stable-diffusion-2-1-base"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

# Move model to GPU
pipe = pipe.to("cuda")

# Generate an image
prompt = "A futuristic cityscape at sunset"
image = pipe(prompt).images[0]

# Save the image
image.save("output.png")
