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

DESCRIPTION = '# [ModelScope Text to Video Synthesis](https://modelscope.cn/models/damo/text-to-video-synthesis/summary)'
DESCRIPTION += '\n<p>This model can only be used for non-commercial purposes. To learn more about the model, take a look at the <a href="https://huggingface.co/damo-vilab/modelscope-damo-text-to-video-synthesis" style="text-decoration: underline;" target="_blank">model card</a>.</p>'

MAX_NUM_FRAMES = int(os.getenv('MAX_NUM_FRAMES', '200'))
DEFAULT_NUM_FRAMES = min(MAX_NUM_FRAMES,
                         int(os.getenv('DEFAULT_NUM_FRAMES', '16')))

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


examples = [
    ['An astronaut riding a horse.', 0, 16, 25],
    ['A panda eating bamboo on a rock.', 0, 16, 25],
    ['Spiderman is surfing.', 0, 16, 25],
]

with gr.Blocks(css='style.css') as demo:
    gr.Markdown(DESCRIPTION)
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
    gr.Examples(examples=examples,
                inputs=inputs,
                outputs=result,
                fn=generate,
                cache_examples=os.getenv('SYSTEM') == 'spaces')

    prompt.submit(fn=generate, inputs=inputs, outputs=result)
    run_button.click(fn=generate, inputs=inputs, outputs=result)
    
    with gr.Accordion(label='Biases and content acknowledgment', open=False):
        gr.HTML("""<div class="acknowledgments">
                    <h4>Biases and content acknowledgment</h4>
                    <p>
                        Despite how impressive being able to turn text into video is, beware to the fact that this model may output content that reinforces or exacerbates societal biases. The training data includes LAION5B, ImageNet, Webvid and other public datasets. The model was not trained to realistically represent people or events, so using it to generate such content is beyond the model's capabilities.
                    </p>
                    <p>
                        It is not intended to generate content that is demeaning or harmful to people or their environment, culture, religion, etc. Similarly, it is not allowed to generate pornographic, violent and bloody content generation. <b>The model is meant for research purposes</b>.
                    </p>
                    <p>
                        To learn more about the model, head to its <a href="https://huggingface.co/damo-vilab/modelscope-damo-text-to-video-synthesis" style="text-decoration: underline;" target="_blank">model card</a>.
                    </p>
                   </div>
                """)

        
demo.queue(api_open=False, max_size=15).launch(share=True)
