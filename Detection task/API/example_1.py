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
    classes = {0: 'Нейросеть 🤖', 1: "Человек 👤"}
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
    pretrained_weights_path: str = 'Detection task/pretrained/NeuroMelNet.pth'

    net = load_checkpoint(load_path=pretrained_weights_path, 
                      model=NeuroMelNet().to(consts.DEVICE), 
                      pretrained_eval=True)
    
    input_file_path: str = input('>>> Write file path: ')
    print(predict_fake_input_audio(input_file_path))
