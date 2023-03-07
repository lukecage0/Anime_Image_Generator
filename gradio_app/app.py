import os

import gradio as gr
import torch

from infer import InferencePipe

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = InferencePipe(path="./trained_models/generator_f_100.pt", device=device)


def inference(image):
    os.system("mkdir test")
    image.save("test/input.jpg", "JPEG")
    pipe.convertImage(path="test/input.jpg", save=True, return_original=False)
    return "fake.png"


title = "AnimeGANv2"
description = "Gradio Demo for AnimeGAN-V1. To use it, simply upload your image, or click one of the examples to load them. Read more at the links below. I would recommend don't input very high res images like 7k use a lower resolution like images in the examples. If you like please do give a star ⭐ to the github repo whose link is mentioned below."
article = "<p style='text-align: center; font-weight: bold;'><a href='https://github.com/Atharva-Phatak/AnimeGAN' target='_blank'>Github Repo 🚀</a></p></p>"
examples = [
    "./images/japan-satoshi.jpg",
    "./images/japan-street.jpg",
    "./images/japan-tower.jpg",
    "./images/Japan.jpg",
]

gr.Interface(
    inference,
    [gr.inputs.Image(type="pil", label="Input")],
    gr.outputs.Image(type="file", label="Output"),
    title=title,
    description=description,
    article=article,
    examples=examples,
    allow_flagging=False,
    allow_screenshot=False,
).launch()
