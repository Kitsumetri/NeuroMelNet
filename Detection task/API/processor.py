import numpy as np

import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset
from typing import Tuple
import pandas as pd

from utils import load_audio, mel_spectrogram, AudioProcessorConfig

class AudioProcessor:
    """
    Processes audio files into mel spectrograms.

    Args:
        sample_rate (int): The sampling rate of the audio files.
        size (int): Desired size of the output mel spectrogram.
    """

    def __init__(self, config: AudioProcessorConfig):
        self.sample_rate = config.sample_rate
        self.size = config.size
        self.n_fft = config.n_fft
        self.hop_length = config.hop_length
        self.n_mels = config.n_mels

    def __call__(self, file_path: str) -> torch.Tensor:
        """
        Processes an audio file into a mel spectrogram tensor.

        Args:
            file_path (str): Path to the audio file.

        Returns:
            torch.Tensor: Mel spectrogram tensor.
        """
        y, sr = load_audio(file_path, sample_rate=self.sample_rate)
        mel_spec = mel_spectrogram(y, sr, 
                                   self.n_fft, self.hop_length, self.n_mels)
        
        mel_spec_tensor = torch.tensor(mel_spec[np.newaxis, np.newaxis, ...], dtype=torch.float32)
        mel_spec_tensor = mel_spec_tensor.squeeze(0)
        mel_spec_tensor = mel_spec_tensor.unsqueeze(0)
        mel_spec_tensor = F.interpolate(mel_spec_tensor, size=self.size)
        
        return mel_spec_tensor

class AudioDataset(Dataset):
    """
    Dataset class for loading audio data with corresponding labels.

    Args:
        metadata (DataFrame): Metadata containing file paths and labels.
        transform (callable, optional): Optional transform to be applied to the audio data.
    """

    def __init__(self, metadata: pd.DataFrame, transform: AudioProcessor):
        self.metadata = metadata
        self.transform = transform
        self.data = []
        self.labels = []
        
        self.load_data()

    def load_data(self):
        """
        Loads audio data and corresponding labels from metadata.
        """
        for idx in tqdm(range(len(self.metadata))):
            file_path = self.metadata.iloc[idx]['file']
            label = self.metadata.iloc[idx]['label']
            
            mel_spec_tensor = self.transform(file_path)
            
            self.data.append(mel_spec_tensor)
            self.labels.append(label)

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Retrieves a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: Tuple containing the mel spectrogram tensor and its label.
        """
        mel_spec_tensor = self.data[idx]
        label = self.labels[idx]
        
        return mel_spec_tensor, label