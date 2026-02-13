from torch.utils.data import DataLoader
from models.dataset import LJSpeechDataset, collate_fn


def main():

    dataset = LJSpeechDataset()

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,   # 👈 IMPORTANT for Windows
        collate_fn=collate_fn
    )

    batch = next(iter(loader))

    print("Batch mel shape:", batch["mel"].shape)
    print("IDs:", batch["ids"])
    print("dtype:", batch["mel"].dtype)


if __name__ == "__main__":
    main()
