# NeuroMelNet

## Quickstart
``` 
git clone https://github.com/Kitsumetri/NeuroMelNet.git
cd NeuroMelNet
conda create --name NeuroMelNetEnv python=3.10
conda activate NeuroMelNetEnv
pip install -r requirements.txt
cd 'Detection task'/API
```

## Examples:
### Predict
```
from torchvision import transforms

import torch

import constants as consts
from model import NeuroMelNet
from processor import AudioProcessor
from utils import load_checkpoint, AudioProcessorConfig

from typing import Tuple


preprocessor = transforms.Compose(
    [
        AudioProcessor(AudioProcessorConfig())
    ]
)


def predict_fake_input_audio(file_path: str) -> Tuple[str, torch.Tensor]:
    classes = {0: 'AI 🤖', 1: "Human 👤"}
    net.eval()

    try:
        mel_tensor = preprocessor(file_path).to(consts.DEVICE)
    except FileNotFoundError:
        return None

    with torch.no_grad():
        outputs = net(mel_tensor.squeeze())
        predicted = torch.argmax(outputs, dim=1).cpu().numpy().item()
        return classes[predicted], torch.round(outputs, decimals=4)


if __name__ == '__main__':
    pretrained_weights_path: str = 'PATH_TO_NeuroMelNet.pth'

    net = load_checkpoint(load_path=pretrained_weights_path, 
                      model=NeuroMelNet().to(consts.DEVICE), 
                      pretrained_eval=True)
    
    input_file_path: str = input('>>> Write file path: ')
    print(predict_fake_input_audio(input_file_path))
```

### Train
```
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
```