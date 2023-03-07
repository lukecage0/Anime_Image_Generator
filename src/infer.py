import argparse
import glob
import time

import cv2
import numpy as np
import torch
from tqdm import tqdm

from modelling.generator import Generator


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_path", required=True, description="Path to the source image"
    )
    parser.add_argument(
        "--dest_path",
        required=True,
        description="Path where generated image should be saved",
    )
    parser.add_argument(
        "--checkpoint_path", required=True, description="Path to ckpt file"
    )
    return parser


def divisible(dim):
    """
    Make width and height divisible by 32
    """
    width, height = dim
    return width - (width % 32), height - (height % 32)


def resize_image(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    h, w = image.shape[:2]

    if width and height:
        return cv2.resize(image, divisible((width, height)), interpolation=inter)

    if width is None and height is None:
        return cv2.resize(image, divisible((w, h)), interpolation=inter)

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, divisible(dim), interpolation=inter)


class InferencePipe:
    def __init__(self, path, device):
        self.device = device
        self.load_model(path)

    def load_model(self, path):
        ckpt = torch.load(path)
        self.generator = Generator()
        self.generator.load_state_dict(ckpt["model_state_dict"])
        self.generator.eval()
        self.generator.to(self.device)

    # self.generator.half()

    @staticmethod
    def to_numpy(tensor):
        """Convert torch tensor to numpy."""
        return (
            tensor.detach().cpu().numpy()
            if tensor.requires_grad
            else tensor.cpu().numpy()
        )

    @staticmethod
    def normalize(image):
        return image / 127.5 - 1.0

    def preprocess_image(self, image):
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

    @torch.no_grad()
    def convertToAnime(self, image):
        image = self.preprocess_image(image)
        image = image.to(self.device)
        # image = image.half()
        fake = self.generator(image)
        fake = fake.type_as(image).detach().cpu().numpy()
        fake = fake.transpose(0, 2, 3, 1)
        return fake

    def convertImage(
        self,
        path=None,
        image=None,
        width=None,
        height=None,
        save=True,
        dtype=None,
        return_original: bool = False,
        save_filename: str = "fake.png",
    ):
        if image is not None:
            image = image[:, :, ::-1]
        else:
            image = cv2.imread(path)[:, :, ::-1]
        image = resize_image(image, width, height)
        fake = self.convertToAnime(image)
        fake = self.denormalize(fake[0], dtype=dtype)
        if save:
            cv2.imwrite(save_filename, fake[..., ::-1])
        else:
            if return_original:
                return image, fake[..., ::-1]

    def writeFrames(self, frame, writer, count):
        # print(frame.shape)
        images = self.convertToAnime(image=frame)
        images = self.denormalize(images, np.uint8)
        for i in range(0, count):
            # img = np.clip(images[i], 0, 255)
            writer.write_frame(images[i][..., ::-1])

    def convertVideo(self, path, output_path, batch_size=2):
        capture = cv2.VideoCapture(path)
        frame_width = int(capture.get(3))
        frame_height = int(capture.get(4))
        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            30,
            (frame_width, frame_height),
        )
        frame_count = total_fps = 0
        while capture.isOpened():
            ret, frame = capture.read()
            if ret:
                start_time = time.time()
                image = (
                    self.convertImage(
                        image=frame, width=frame_width, height=frame_height, save=False
                    )
                    / 255
                )
                end_time = time.time()
                fps = 1 / (end_time - start_time)
                # add `fps` to `total_fps`
                total_fps += fps
                # increment frame count
                image = np.uint8(image)
                frame_count += 1
                out.write(image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break
        capture.release()
        # close all frames and video windows
        cv2.destroyAllWindows()
        # calculate and print the average FPS
        avg_fps = total_fps / frame_count
        print(f"Average FPS: {avg_fps:.3f}")


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = InferencePipe(path=args.checkpoint_path, device=device)
    pipe.convertImage(path=args.source_path, save_filename=args.dest_path)
