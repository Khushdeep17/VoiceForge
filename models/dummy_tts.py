import torch
import torch.nn as nn


class DummyTTS(nn.Module):
    """
    Very small model.
    Just validates the pipeline.
    """

    def __init__(self, n_mels=80):

        super().__init__()

        self.net = nn.Sequential(

            nn.Conv1d(n_mels, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv1d(128, n_mels, kernel_size=3, padding=1)

        )

    def forward(self, mel):
        return self.net(mel)
