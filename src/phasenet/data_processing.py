from seisbench.data import WaveformDataset
import seisbench.generate as sbg
from torch.utils.data import DataLoader
import numpy as np


def load_data(path: str) -> WaveformDataset:
    ds = WaveformDataset(path)
    return ds


def split_data(ds: WaveformDataset):
    train_ds = ds.train()
    dev_ds = ds.dev()
    test_ds = ds.test()

    return train_ds, dev_ds, test_ds


def create_generator(ds: WaveformDataset, augmentations: list) -> sbg.GenericGenerator:
    generator = sbg.GenericGenerator(ds)

    generator.add_augmentations(augmentations)
    return generator


def create_dataloader(generator: sbg.GenericGenerator, batch_size: int, num_workers: int) -> DataLoader:
    dataloader = DataLoader(
        generator,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    return dataloader


def create_augmentations(phase_dict: dict, model) -> list:
    augmentations = [
        sbg.WindowAroundSample(
            list(phase_dict.keys()),
            samples_before=3000,
            windowlen=6000,
            selection="random",
            strategy="variable",
        ),
        sbg.RandomWindow(windowlen=3001, strategy="pad"),
        sbg.ChangeDtype(np.float32),
        sbg.ProbabilisticLabeller(
            label_columns=phase_dict, model_labels=model.labels, sigma=30, dim=0
        ),
        sbg.ChangeDtype(np.float32, key="y"),
    ]
    return augmentations