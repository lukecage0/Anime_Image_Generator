import base64
import io
import os
import urllib
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from pathlib import Path

import lightning as L
import torch
from animeGAN.constants import AppConstants, AppDescription
from animeGAN.pipeline import InferencePipeline
from animeGAN.utils import TimeoutException
from fastapi import File, Response, UploadFile
from PIL import Image
from starlette.responses import RedirectResponse


@dataclass
class AnimeGANConfig(L.BuildConfig):
    requirements = ["fastapi==0.78.0", "uvicorn==0.17.6", "torch", "numpy"]


class AnimeGANServe(L.LightningWork):
    def __init__(self, **kwargs):
        super().__init__(cloud_build_config=AnimeGANConfig(), **kwargs)
        self._model = None
        self.api_url = ""

    @staticmethod
    def _download_weights(url: str, storePath: str):
        dest = storePath / f"generator.pt"
        if not os.path.exists(dest):
            urllib.request.urlretrieve(url, dest)

    def build_pipeline(self):
        fp16 = True if torch.cuda.is_available() else False
        device = "cuda" if fp16 else "cpu"
        weights_path = Path("resources/trained_models")
        weights_path.mkdir(parents=True, exist_ok=True)
        self._download_weights(
            url="https://github.com/Atharva-Phatak/AnimeGAN/releases/download/0.0.1/generator_f_100.pt",
            storePath=weights_path,
        )
        self._model = InferencePipeline(
            weights_path=weights_path, device=device, use_fp16=fp16
        )

    def predict(self, data: bytes):
        image = Image.open(io.BytesIO(data))
        generatedImage = self._model.convertToAnime(image)
        buffered = io.BytesIO()
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return Response(content=img_str, media_type="image/png")

    def run(self):
        import subprocess

        import uvicorn
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware

        if self._model is None:
            self.build_pipeline()

        app = FastAPI(title="Backend for AnimeGAN-v2", description=AppDescription.desc)
        app.POOL: ThreadPoolExecutor = None

        @app.get("/", include_in_schema=False)
        async def index():
            return RedirectResponse(url="/docs")

        @app.on_event("startup")
        def startup_event():
            app.POOL = ThreadPoolExecutor(max_workers=1)

        @app.on_event("shutdown")
        def shutdown_event():
            app.POOL.shutdown(wait=False)

        @app.get("/api/health")
        def health():
            return True

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.post("/api/predict")
        async def predict_api(data: UploadFile = File(...)):
            try:
                data = await data.read()
                result = app.POOL.submit(self.predict, data).result()
                return result
            except (TimeoutError, TimeoutException):
                raise TimeoutException()

        uvicorn.run(
            app,
            timeout_keep_alive=AppConstants.KEEP_ALIVE_TIMEOUT,
            access_log=False,
            loop="uvloop",
            host=self.host,
            port=self.port,
        )
