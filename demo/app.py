import gradio as gr
import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline, LCMScheduler
from diffusers.schedulers import TCDScheduler

import spaces
from PIL import Image

SAFETY_CHECKER = True

checkpoints = {
    "2-Step": ["pcm_{}_smallcfg_2step_converted.safetensors", 2, 0.0],
    "4-Step": ["pcm_{}_smallcfg_4step_converted.safetensors", 4, 0.0],
    "8-Step": ["pcm_{}_smallcfg_8step_converted.safetensors", 8, 0.0],
    "16-Step": ["pcm_{}_smallcfg_16step_converted.safetensors", 16, 0.0],
    "Normal CFG 4-Step": ["pcm_{}_normalcfg_4step_converted.safetensors", 4, 7.5],
    "Normal CFG 8-Step": ["pcm_{}_normalcfg_8step_converted.safetensors", 8, 7.5],
    "Normal CFG 16-Step": ["pcm_{}_normalcfg_16step_converted.safetensors", 16, 7.5],
    "LCM-Like LoRA": [
        "pcm_{}_lcmlike_lora_converted.safetensors",
        4,
        0.0,
    ],
}


loaded = None

if torch.cuda.is_available():
    pipe_sdxl = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
    ).to("cuda")
    pipe_sd15 = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16"
    ).to("cuda")

if SAFETY_CHECKER:
    from safety_checker import StableDiffusionSafetyChecker
    from transformers import CLIPFeatureExtractor

    safety_checker = StableDiffusionSafetyChecker.from_pretrained(
        "CompVis/stable-diffusion-safety-checker"
    ).to("cuda")
    feature_extractor = CLIPFeatureExtractor.from_pretrained(
        "openai/clip-vit-base-patch32"
    )

    def check_nsfw_images(
        images: list[Image.Image],
    ) -> tuple[list[Image.Image], list[bool]]:
        safety_checker_input = feature_extractor(images, return_tensors="pt").to("cuda")
        has_nsfw_concepts = safety_checker(
            images=[images], clip_input=safety_checker_input.pixel_values.to("cuda")
        )

        return images, has_nsfw_concepts


@spaces.GPU(enable_queue=True)
def generate_image(
    prompt,
    ckpt,
    num_inference_steps,
    progress=gr.Progress(track_tqdm=True),
    mode="sdxl",
):
    global loaded
    checkpoint = checkpoints[ckpt][0].format(mode)
    guidance_scale = checkpoints[ckpt][2]
    pipe = pipe_sdxl if mode == "sdxl" else pipe_sd15

    if loaded != (ckpt + mode):
        pipe.load_lora_weights(
            "wangfuyun/PCM_Weights", weight_name=checkpoint, subfolder=mode
        )
        loaded = ckpt + mode

        if ckpt == "LCM-Like LoRA":
            pipe.scheduler = LCMScheduler()
        else:
            pipe.scheduler = TCDScheduler(
                num_train_timesteps=1000,
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                timestep_spacing="trailing",
            )

    results = pipe(
        prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale
    )

    if SAFETY_CHECKER:
        images, has_nsfw_concepts = check_nsfw_images(results.images)
        if any(has_nsfw_concepts):
            gr.Warning("NSFW content detected.")
            return Image.new("RGB", (512, 512))
        return images[0]
    return results.images[0]


def update_steps(ckpt):
    num_inference_steps = checkpoints[ckpt][1]
    if ckpt == "LCM-Like LoRA":
        return gr.update(interactive=True, value=num_inference_steps)
    return gr.update(interactive=False, value=num_inference_steps)


css = """
.gradio-container {
  max-width: 60rem !important;
}
"""
with gr.Blocks(css=css) as demo:
    gr.Markdown(
        """
# Phased Consistency Model  

Phased Consistency Model (PCM) is an image generation technique that addresses the limitations of the Latent Consistency Model (LCM) in high-resolution and text-conditioned image generation.
PCM outperforms LCM across various generation settings and achieves state-of-the-art results in both image and video generation.

[[paper](https://huggingface.co/papers/2405.18407)] [[arXiv](https://arxiv.org/abs/2405.18407)]  [[code](https://github.com/G-U-N/Phased-Consistency-Model)] [[project page](https://g-u-n.github.io/projects/pcm)]
"""
    )
    with gr.Group():
        with gr.Row():
            prompt = gr.Textbox(label="Prompt", scale=8)
            ckpt = gr.Dropdown(
                label="Select inference steps",
                choices=list(checkpoints.keys()),
                value="4-Step",
            )
            steps = gr.Slider(
                label="Number of Inference Steps",
                minimum=1,
                maximum=20,
                step=1,
                value=4,
                interactive=False,
            )
            ckpt.change(
                fn=update_steps,
                inputs=[ckpt],
                outputs=[steps],
                queue=False,
                show_progress=False,
            )

            submit_sdxl = gr.Button("Run on SDXL", scale=1)
            submit_sd15 = gr.Button("Run on SD15", scale=1)

    img = gr.Image(label="PCM Image")
    gr.Examples(
        examples=[
            [" astronaut walking on the moon", "4-Step", 4],
            [
                "Photo of a dramatic cliffside lighthouse in a storm, waves crashing, symbol of guidance and resilience.",
                "8-Step",
                8,
            ],
            [
                "Vincent vangogh style, painting, a boy, clouds in the sky",
                "Normal CFG 4-Step",
                4,
            ],
            [
                "Echoes of a forgotten song drift across the moonlit sea, where a ghost ship sails, its spectral crew bound to an eternal quest for redemption.",
                "4-Step",
                4,
            ],
            [
                "Roger rabbit as a real person, photorealistic, cinematic.",
                "16-Step",
                16,
            ],
            [
                "tanding tall amidst the ruins, a stone golem awakens, vines and flowers sprouting from the crevices in its body.",
                "LCM-Like LoRA",
                4,
            ],
        ],
        inputs=[prompt, ckpt, steps],
        outputs=[img],
        fn=generate_image,
        cache_examples="lazy",
    )

    gr.on(
        fn=generate_image,
        triggers=[ckpt.change, prompt.submit, submit_sdxl.click],
        inputs=[prompt, ckpt, steps],
        outputs=[img],
    )
    gr.on(
        fn=lambda *args: generate_image(*args, mode="sd15"),
        triggers=[submit_sd15.click],
        inputs=[prompt, ckpt, steps],
        outputs=[img],
    )


demo.queue(api_open=False).launch(show_api=False)
