#!/usr/bin/env python

from __future__ import annotations

import os
import pathlib
import random
import shlex
import subprocess

import gradio as gr
import torch
from huggingface_hub import snapshot_download

if os.getenv('SYSTEM') == 'spaces':
    subprocess.run(shlex.split('pip uninstall -y modelscope'))
    subprocess.run(
        shlex.split(
            'pip install git+https://github.com/modelscope/modelscope.git@refs/pull/207/head'
        ))

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline

model_dir = pathlib.Path('weights')
if not model_dir.exists():
    model_dir.mkdir()
    snapshot_download('hysts/modelscope-damo-text-to-video-synthesis',
                      repo_type='model',
                      local_dir=model_dir)

DESCRIPTION = '# [ModelScope Text to Video Synthesis](https://modelscope.cn/models/damo/text-to-video-synthesis/summary)'
if (SPACE_ID := os.getenv('SPACE_ID')) is not None:
    DESCRIPTION += f'\n<p>For faster inference without waiting in queue, you may duplicate the space and upgrade to GPU in settings. <a href="https://huggingface.co/spaces/{SPACE_ID}?duplicate=true"><img style="display: inline; margin-top: 0em; margin-bottom: 0em" src="https://bit.ly/3gLdBN6" alt="Duplicate Space" /></a></p>'

pipe = pipeline('text-to-video-synthesis', model_dir.as_posix())


def generate(prompt: str, seed: int) -> str:
    if seed == -1:
        seed = random.randint(0, 1000000)
    torch.manual_seed(seed)
    return pipe({'text': prompt})[OutputKeys.OUTPUT_VIDEO]


examples = [
    ['An astronaut riding a horse.', 0],
    ['A panda eating bamboo on a rock.', 0],
    ['Spiderman is surfing.', 0],
]

with gr.Blocks(css='style.css') as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column():
            prompt = gr.Text(label='Prompt', max_lines=1)
            seed = gr.Slider(
                label='Seed',
                minimum=-1,
                maximum=1000000,
                step=1,
                value=-1,
                info='If set to -1, a different seed will be used each time.')
            run_button = gr.Button('Run')
        with gr.Column():
            result = gr.Video(label='Result')

    inputs = [prompt, seed]
    gr.Examples(examples=examples,
                inputs=inputs,
                outputs=result,
                fn=generate,
                cache_examples=os.getenv('SYSTEM') == 'spaces')

    prompt.submit(fn=generate, inputs=inputs, outputs=result)
    run_button.click(fn=generate, inputs=inputs, outputs=result)

demo.queue(api_open=False).launch()
