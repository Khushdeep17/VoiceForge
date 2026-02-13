import yaml
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from models.dataset import LJSpeechDataset, collate_fn


def load_config(path="configs/local.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_loaders(config_path="configs/local.yaml"):

    config = load_config(config_path)

    dataset = LJSpeechDataset()

    indices = list(range(len(dataset)))

    train_idx, val_idx = train_test_split(
        indices,
        train_size=config["train_split"],
        random_state=config["seed"],
        shuffle=True
    )

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        collate_fn=collate_fn,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        collate_fn=collate_fn,
        pin_memory=False
    )

    return train_loader, val_loader


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()

    train_loader, val_loader = build_loaders()

    batch = next(iter(train_loader))

    print("Train batch mel shape:", batch["mel"].shape)
    print("Validation batches:", len(val_loader))
