from dataclasses import dataclass


@dataclass
class AppConstants:
    KEEP_ALIVE_TIMEOUT: int = 160
    INFERENCE_REQUEST_TIMEOUT: int = 160


@dataclass
class AppDescription:
    desc: str = """<h2>Try this app by uploading any image with `/api/predict` and get an anime style image.</h2>
<h2>For training code please visit : https://github.com/Atharva-Phatak/AnimeGAN</h2>
<br>by Atharva Phatak"""
