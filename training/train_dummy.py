import torch
import torch.nn as nn
from tqdm import tqdm
import yaml
import time
from pathlib import Path

from models.dummy_tts import DummyTTS
from training.build_dataloaders import build_loaders


def load_config(path="configs/local.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def train():

    config = load_config()

    device = torch.device(config["device"])

    # Ensure folders exist (pro move)
    Path("checkpoints").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    train_loader, val_loader = build_loaders()

    model = DummyTTS().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    criterion = nn.L1Loss()

    epochs = 3

    print("Starting sanity training...\n")

    start_time = time.time()

    for epoch in range(epochs):

        # -------------------
        # TRAINING
        # -------------------
        model.train()
        running_loss = 0

        loop = tqdm(train_loader)

        for batch in loop:

            mel = batch["mel"].to(device)

            pred = model(mel)

            loss = criterion(pred, mel)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            loop.set_description(f"Epoch {epoch+1}")
            loop.set_postfix(train_loss=loss.item())

        avg_train_loss = running_loss / len(train_loader)

        # -------------------
        # VALIDATION (high signal)
        # -------------------
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                mel = batch["mel"].to(device)
                pred = model(mel)
                loss = criterion(pred, mel)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(
            f"\nEpoch {epoch+1} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}"
        )

        # -------------------
        # LOGGING (append mode)
        # -------------------
        with open("logs/dummy_train_log.txt", "a") as f:
            f.write(
                f"Epoch {epoch+1} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f}\n"
            )

    # -------------------
    # SAVE CHECKPOINT (timestamped)
    # -------------------
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    checkpoint_path = f"checkpoints/dummy_tts_{timestamp}.pt"

    torch.save(model.state_dict(), checkpoint_path)

    total_time = time.time() - start_time

    with open("logs/dummy_train_log.txt", "a") as f:
        f.write(f"\nTraining Time: {total_time/60:.2f} minutes\n")
        f.write(f"Checkpoint: {checkpoint_path}\n\n")

    print("\n✅ Training complete.")
    print(f"Model saved to {checkpoint_path}")
    print(f"Total training time: {total_time/60:.2f} minutes")


if __name__ == "__main__":
    train()
