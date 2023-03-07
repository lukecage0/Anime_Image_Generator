import numpy as np
import torch
from animeGAN.modelling import Generator
from animeGAN.modelling.utils import resize_image
from PIL import Image


class InferencePipeline:
    def __init__(self, weights_path: str, device: str, use_fp16: bool = True):
        self.fp16 = use_fp16
        self.device = device
        self.load_model(weights_path)
        # self.load_model(path=weights_path)

    def load_model(self, path):
        map_location = torch.device("cpu") if self.device == "cpu" else None
        ckpt = torch.load(path / "generator.pt", map_location=map_location)
        self._generatorModel = Generator()
        self._generatorModel.load_state_dict(ckpt["model_state_dict"])
        self._generatorModel.eval().to(self.device)
        if self.fp16:
            self._generatorModel.half()

    @staticmethod
    def normalize(image):
        return image / 127.5 - 1.0

    @staticmethod
    def denormalize(image, dtype=None):

        image = image * 127.5 + 127.5

        if dtype is not None:
            if isinstance(image, torch.Tensor):
                image = image.type(dtype)
            else:
                # numpy.ndarray
                image = image.astype(dtype)

        return image

    def preprocess_image(self, image):
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        image = image.astype(np.float32)
        # Normalize to [-1, 1]
        image = self.normalize(image)
        image = torch.from_numpy(image)
        # Add batch dim
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        # image = image.half()
        # channel first
        image = image.permute(0, 3, 1, 2)
        return image

    @torch.inference_mode()
    def forwardPass(self, image):
        image = self.preprocess_image(image)
        if self.fp16:
            image = image.half()
        image = image.to(self.device)
        fake = self._generatorModel(image)
        fake = fake.type_as(image).detach().cpu().numpy()
        fake = fake.transpose(0, 2, 3, 1)
        return fake

    def convertToAnime(self, image):
        fake = self.forwardPass(image)
        # dtype = np.uint8 if self.fp16 else None
        fake = Image.fromarray(self.denormalize(fake[0], dtype=np.uint8))
        return fake
