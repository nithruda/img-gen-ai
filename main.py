use_refiner = False

import mediapy as media
import random
import sys
import torch

from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
    )

if use_refiner:
  refiner = DiffusionPipeline.from_pretrained(
      "stabilityai/stable-diffusion-xl-refiner-1.0",
      text_encoder_2=pipe.text_encoder_2,
      vae=pipe.vae,
      torch_dtype=torch.float16,
      use_safetensors=True,
      variant="fp16",
  )

  refiner = refiner.to("cuda")

  pipe.enable_model_cpu_offload()
else:
  pipe = pipe.to("cuda")

  prompt = input("Enter a prompt: ")

  # Rest of your code
  seed = random.randint(0, sys.maxsize)

  # The rest of your code remains the same
  images = pipe(
      prompt=prompt,
      output_type="latent" if use_refiner else "pil",
      generator=torch.Generator("cuda").manual_seed(seed),
  ).images

  if use_refiner:
      images = refiner(
          prompt=prompt,
          image=images,
      ).images

  print(f"Prompt:\t{prompt}\nSeed:\t{seed}")
  media.show_images(images)
  images[0].save("output.jpg")