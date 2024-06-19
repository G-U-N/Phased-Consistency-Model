import torch
from diffusers import StableDiffusion3Pipeline
from safetensors.torch import load_file, save_file
from diffusers.utils import convert_unet_state_dict_to_peft
from peft import LoraConfig, set_peft_model_state_dict
import numpy as np
from PIL import Image
from pcm_fm_deterministic_scheduler import PCMFMDeterministicScheduler
from pcm_fm_stochastic_scheduler import PCMFMStochasticScheduler

pipe = StableDiffusion3Pipeline.from_pretrained(
    "[PATH TO SD3]", scheduler=PCMFMStochasticScheduler(1000, 3, 50)
)
pcm_lora_weight = load_file("pcm_stochastic_shift3.safetensors")
alpha = 1.0
pcm_lora_weight = {
    key: value * np.sqrt(alpha) for key, value in pcm_lora_weight.items()
}
pipe.load_lora_weights(pcm_lora_weight)
pipe = pipe.to("cuda")

prompt = "portrait photo of a girl, photograph, highly detailed face, depth of field, moody light, golden hour, style by Dan Winters, Russell James, Steve McCurry, centered, extremely detailed, Nikon D850, award winning photography"


with torch.autocast("cuda"):
    images = pipe(
        prompt=prompt,
        negative_prompt="",
        num_inference_steps=4,
        guidance_scale=1.2,
        num_images_per_prompt=9,
    ).images

width, height = images[0].size

grid_width = 3
grid_height = 3
result_image = Image.new("RGB", (grid_width * width, grid_height * height))

for idx, image in enumerate(images):
    x = (idx % grid_width) * width
    y = (idx // grid_width) * height
    result_image.paste(image, (x, y))

result_image.save(prompt[:5] + prompt[-5:] + ".png")


pipe = StableDiffusion3Pipeline.from_pretrained(
    "[PATH TO SD3]", scheduler=PCMFMStochasticScheduler(1000, 3, 50)
)
pcm_lora_weight = load_file("pcm_stochastic_shift3.safetensors")
alpha = 1.0
pcm_lora_weight = {
    key: value * np.sqrt(alpha) for key, value in pcm_lora_weight.items()
}
pipe.load_lora_weights(pcm_lora_weight)
pipe = pipe.to("cuda")

prompt = "portrait photo of a girl, photograph, highly detailed face, depth of field, moody light, golden hour, style by Dan Winters, Russell James, Steve McCurry, centered, extremely detailed, Nikon D850, award winning photography"


with torch.autocast("cuda"):
    images = pipe(
        prompt=prompt,
        negative_prompt="",
        num_inference_steps=4,
        guidance_scale=1.2,
        num_images_per_prompt=9,
    ).images

width, height = images[0].size

grid_width = 3
grid_height = 3
result_image = Image.new("RGB", (grid_width * width, grid_height * height))

for idx, image in enumerate(images):
    x = (idx % grid_width) * width
    y = (idx // grid_width) * height
    result_image.paste(image, (x, y))

result_image.save(prompt[:5] + prompt[-5:] + "stochastic" + ".png")


pipe = StableDiffusion3Pipeline.from_pretrained(
    "[PATH TO SD3]", scheduler=PCMFMDeterministicScheduler(1000, 1, 50)
)
pcm_lora_weight = load_file("pcm_deterministic_2step_shift1.safetensors")
alpha = 1.0
pcm_lora_weight = {
    key: value * np.sqrt(alpha) for key, value in pcm_lora_weight.items()
}
pipe.load_lora_weights(pcm_lora_weight)
pipe = pipe.to("cuda")

prompt = "portrait photo of a girl, photograph, highly detailed face, depth of field, moody light, golden hour, style by Dan Winters, Russell James, Steve McCurry, centered, extremely detailed, Nikon D850, award winning photography"


with torch.autocast("cuda"):
    images = pipe(
        prompt=prompt,
        negative_prompt="",
        num_inference_steps=4,
        guidance_scale=1.2,
        num_images_per_prompt=9,
    ).images

width, height = images[0].size

grid_width = 3
grid_height = 3
result_image = Image.new("RGB", (grid_width * width, grid_height * height))

for idx, image in enumerate(images):
    x = (idx % grid_width) * width
    y = (idx // grid_width) * height
    result_image.paste(image, (x, y))

result_image.save(prompt[:5] + prompt[-5:] + "2step_shift3" + ".png")


pipe = StableDiffusion3Pipeline.from_pretrained(
    "[PATH TO SD3]", scheduler=PCMFMDeterministicScheduler(1000, 3, 50)
)
pcm_lora_weight = load_file("pcm_deterministic_4step_shift3.safetensors")
alpha = 1.0
pcm_lora_weight = {
    key: value * np.sqrt(alpha) for key, value in pcm_lora_weight.items()
}
pipe.load_lora_weights(pcm_lora_weight)
pipe = pipe.to("cuda")

prompt = "portrait photo of a girl, photograph, highly detailed face, depth of field, moody light, golden hour, style by Dan Winters, Russell James, Steve McCurry, centered, extremely detailed, Nikon D850, award winning photography"


with torch.autocast("cuda"):
    images = pipe(
        prompt=prompt,
        negative_prompt="",
        num_inference_steps=4,
        guidance_scale=1.2,
        num_images_per_prompt=9,
    ).images

width, height = images[0].size

grid_width = 3
grid_height = 3
result_image = Image.new("RGB", (grid_width * width, grid_height * height))

for idx, image in enumerate(images):
    x = (idx % grid_width) * width
    y = (idx // grid_width) * height
    result_image.paste(image, (x, y))

result_image.save(prompt[:5] + prompt[-5:] + "4step_shift3" + ".png")


pipe = StableDiffusion3Pipeline.from_pretrained(
    "[PATH TO SD3]", scheduler=PCMFMDeterministicScheduler(1000, 1, 50)
)
pcm_lora_weight = load_file("pcm_deterministic_4step_shift1.safetensors")
alpha = 1.0
pcm_lora_weight = {
    key: value / 2 * np.sqrt(alpha) for key, value in pcm_lora_weight.items()
}
pipe.load_lora_weights(pcm_lora_weight)
pipe = pipe.to("cuda")

prompt = "a girl lay on the grass"


with torch.autocast("cuda"):
    images = pipe(
        prompt=prompt,
        negative_prompt="",
        num_inference_steps=4,
        guidance_scale=1.2,
        num_images_per_prompt=9,
    ).images

width, height = images[0].size

grid_width = 3
grid_height = 3
result_image = Image.new("RGB", (grid_width * width, grid_height * height))

for idx, image in enumerate(images):
    x = (idx % grid_width) * width
    y = (idx // grid_width) * height
    result_image.paste(image, (x, y))

result_image.save(prompt[:5] + prompt[-5:] + "4step_shift1" + ".png")
