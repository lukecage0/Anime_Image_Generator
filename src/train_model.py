import argparse

import pytorch_lightning as pl

# from params import Params
from omegaconf import OmegaConf
from pytorch_lightning.loggers import MLFlowLogger

from train_pl import AnimeGANTrainer


def trainCLI():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    # parser.parse_args()
    return parser


def setupTrainer(path):
    params = OmegaConf.load(path)
    module = AnimeGANTrainer(path)
    mlf_logger = MLFlowLogger(
        experiment_name="animeGan-Training-hayao", save_dir="./ml-runs-hayao"
    )
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=params.epochs,
        auto_select_gpus=True,
        logger=mlf_logger,
    )
    trainer.fit(module)


if __name__ == "__main__":
    parser = trainCLI()
    args = parser.parse_args()
    setupTrainer(args.path)
