import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass, field
from processor import AudioProcessor, AudioDataset
from utils import AudioProcessorConfig
import pandas as pd
from termcolor import colored
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TrainConfig:
    device: torch.device
    batch_size: int
    learning_rate: float
    num_epochs: int
    test_size: float
    random_seed: int
    criterion: nn.Module
    model: nn.Module
    metadata: pd.DataFrame
    output_path: str
    audio_processor_config: AudioProcessorConfig
    optimizer: optim.Optimizer = field(init=False)
    train_dataloader: DataLoader = field(init=False)

    def __post_init__(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

def prepare_datasets(train_config: TrainConfig):
    train_metadata, test_metadata = train_test_split(train_config.metadata, 
                                                     test_size=train_config.test_size, 
                                                     random_state=train_config.random_seed)
    
    transform = transforms.Compose([AudioProcessor(train_config.audio_processor_config)])
    train_dataset = AudioDataset(train_metadata, transform=transform)
    logger.info(colored("Train dataset was loaded!", 'green'))

    test_dataset = AudioDataset(test_metadata, transform=transform)
    logger.info(colored("Test dataset was loaded!", 'green'))

    train_dataloader = DataLoader(train_dataset, batch_size=train_config.batch_size, shuffle=True)
    
    return train_dataloader, test_dataset

def train_model(train_config: TrainConfig):
    model = train_config.model
    model.train()
    optimizer = train_config.optimizer
    criterion = train_config.criterion
    train_dataloader = train_config.train_dataloader
    train_losses = []

    for epoch in range(train_config.num_epochs):
        train_loss = 0.0
        for data, target in tqdm(train_dataloader, desc=f"Epoch [{epoch+1}/{train_config.num_epochs}]"):
            data, target = data[0].to(train_config.device), target[0].to(train_config.device)
            optimizer.zero_grad()
            outputs = model(data.squeeze())
            target = torch.eye(2, device=train_config.device)[target.squeeze()].unsqueeze(0)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_dataloader)
        train_losses.append(train_loss)
        output = f"Epoch [{epoch+1}/{train_config.num_epochs}], Train Loss: {train_loss:.4f}"
        logger.info(colored(output, 'cyan'))
    
    return train_losses

def save_model(model, optimizer, path: str, num_epochs: int):
    try:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': num_epochs
        }, path)
        logger.info(colored(f"Model checkpoint saved at {path}", 'yellow'))
    except RuntimeError:
        os.makedirs(os.path.dirname(path))
        save_model(model, optimizer, path, num_epochs)
        logger.info(colored(f"Save folder was created. Path: {os.path.dirname(path)}", 'yellow'))


def execute_training(train_config: TrainConfig, save_weights: bool = True):
    train_dataloader, _ = prepare_datasets(train_config)
    train_config.train_dataloader = train_dataloader
    
    train_history = train_model(train_config)

    if save_weights:
        save_model(train_config.model, 
                   train_config.optimizer, 
                   train_config.output_path, 
                   train_config.num_epochs)

    return train_history
