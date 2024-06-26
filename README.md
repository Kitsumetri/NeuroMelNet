# NeuroMelNet
## Introduction
**NeuroMelNet** - End-to-End binary classification model for AI speech detection with PyTorch implementation. Python API for predict and training/fine-tuning is provided (check examples). Model is similar to RawNet2, but interpolated mel-spectograms are inputs for this model (see AudioProcessor for more information). In-the-wild & WaveFake datasets were used for training.

Pretrained model weights: https://drive.google.com/file/d/1g65xB2JwEE2gt1aJAqmpYeLdU00k8sLz/view?usp=sharing

![Untitled(5)](https://github.com/Kitsumetri/NeuroMelNet/assets/100523204/c3ba29aa-ef74-4197-b90d-7cd841742ab1)

## Model metrics
|Metric     |value   |
|-----------|--------|
| EER       | 0.0223 |
| Accuracy  |0.9858  |
| Precision | 0.9774 |
| F1 Score  |0.9859  |

![изображение](https://github.com/Kitsumetri/NeuroMelNet/assets/100523204/0ad3adf3-d7c6-4f3d-82a4-56106149bf1c)


## Quickstart
```shell
git clone https://github.com/Kitsumetri/NeuroMelNet.git
cd NeuroMelNet
conda create --name NeuroMelNetEnv python=3.10
conda activate NeuroMelNetEnv
pip install -r requirements.txt
cd 'Detection task'/API
```

## Examples:
### Predict
```python
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
```python
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

## References
```bibitex
@article{jung2019RawNet,
  title={RawNet: Advanced end-to-end deep neural network using raw waveforms for text-independent speaker verification},
  author={Jung, Jee-weon and Heo, Hee-soo and Kim, ju-ho and Shim, Hye-jin and Yu, Ha-jin},
  journal={Proc. Interspeech 2019},
  pages={1268--1272},
  year={2019}
}

@article{jung2020improved,
  title={Improved RawNet with Feature Map Scaling for Text-independent Speaker Verification using Raw Waveforms},
  author={Jung, Jee-weon and Kim, Seung-bin and Shim, Hye-jin and Kim, Ju-ho and Yu, Ha-Jin},
  journal={Proc. Interspeech 2020},
  pages={3583--3587},
  year={2020}
}

@article{muller2022does,
  title={Does audio deepfake detection generalize?},
  author={M{\"u}ller, Nicolas M and Czempin, Pavel and Dieckmann, Franziska and Froghyar, Adam and B{\"o}ttinger, Konstantin},
  journal={Interspeech},
  year={2022}
}

@InProceedings{Sun_2023_CVPR,
    author    = {Sun, Chengzhe and Jia, Shan and Hou, Shuwei and Lyu, Siwei},
    title     = {AI-Synthesized Voice Detection Using Neural Vocoder Artifacts},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2023},
    pages     = {904-912}
}
```
