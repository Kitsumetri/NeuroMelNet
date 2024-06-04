import torch.nn as nn
import pandas as pd

from utils import AudioProcessorConfig
from model import NeuroMelNet
from train import TrainConfig, execute_training
import constants as consts

# metadata format:
# format: .csv
# file: str - path to audio file (mp3 | wav)
# label: str - number of class: 0 - Fake, 1 - True
# ______________
# |file | label|
# |_____|______|
# |0.wav|   1  |
# |1.mp3|   0  |
# |.....|......|
# |____________|


def main():
    net = NeuroMelNet().to(consts.DEVICE)

    train_config = TrainConfig(
        audio_processor_config=AudioProcessorConfig(
            sample_rate=22_000
        ),
        device=consts.DEVICE,
        batch_size=1,
        learning_rate=0.001,
        num_epochs=20,
        test_size=0.2,
        random_seed=42,
        criterion=nn.BCELoss(),
        model=net,
        metadata=pd.read_csv('Detection task/API/metadata_example.csv'),
        output_path='pretrained_folder/NeuroMelNet_v1.pth'
    )

    # train_history - list of train loss via epoch
    train_history = execute_training(train_config, 
                                     save_weights=True)


if __name__ == "__main__":
    main()