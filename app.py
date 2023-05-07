#!/usr/bin/env python

from __future__ import annotations

import os
import random
import tempfile

import gradio as gr
import imageio
import numpy as np
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

MAX_NUM_FRAMES = int(os.getenv('MAX_NUM_FRAMES', '200'))
DEFAULT_NUM_FRAMES = min(MAX_NUM_FRAMES, int(os.getenv('DEFAULT_NUM_FRAMES', '16')))

pipe = DiffusionPipeline.from_pretrained('vdo/text-to-video-ms-1.7b', torch_dtype=torch.float16, safety_checker=None)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()

def to_video(frames: list[np.ndarray], fps: int) -> str:
    out_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    writer = imageio.get_writer(out_file.name, format='FFMPEG', fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    return out_file.name

def generate(prompt: str, seed: int, num_frames: int,
             num_inference_steps: int) -> str:
    if seed == -1:
        seed = random.randint(0, 1000000)
    generator = torch.Generator().manual_seed(seed)
    frames = pipe(prompt,
                  num_inference_steps=num_inference_steps,
                  num_frames=num_frames,
                  generator=generator).frames
    return to_video(frames, 8)

with gr.Blocks(css='style.css') as demo:
    with gr.Group():
        with gr.Box():
            with gr.Row(elem_id='prompt-container').style(equal_height=True):
                prompt = gr.Text(
                    label='Prompt',
                    show_label=False,
                    max_lines=1,
                    placeholder='Enter your prompt',
                    elem_id='prompt-text-input').style(container=False)
                run_button = gr.Button('Generate video').style(
                    full_width=False)
        result = gr.Video(label='Result', show_label=False, elem_id='gallery')
        with gr.Accordion('Advanced options', open=False):
            seed = gr.Slider(
                label='Seed',
                minimum=-1,
                maximum=1000000,
                step=1,
                value=-1,
                info='If set to -1, a different seed will be used each time.')
            num_frames = gr.Slider(
                label='Number of frames',
                minimum=16,
                maximum=MAX_NUM_FRAMES,
                step=1,
                value=16,
                info=
                'Note that the content of the video also changes when you change the number of frames.'
            )
            num_inference_steps = gr.Slider(label='Number of inference steps',
                                            minimum=10,
                                            maximum=50,
                                            step=1,
                                            value=25)

    inputs = [
        prompt,
        seed,
        num_frames,
        num_inference_steps,
    ]

    prompt.submit(fn=generate, inputs=inputs, outputs=result)
    run_button.click(fn=generate, inputs=inputs, outputs=result)
   
demo.queue(api_open=False, max_size=15).launch(share=True)
