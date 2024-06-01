import numpy as np

import torch
import torch.nn.functional as F

from utils import load_audio, mel_spectrogram

class AudioProcessor:
    """
    Processes audio files into mel spectrograms.

    Args:
        sample_rate (int): The sampling rate of the audio files.
        size (int): Desired size of the output mel spectrogram.
    """

    def __init__(self, sample_rate: int, size: int):
        self.sample_rate = sample_rate
        self.size = size

    def __call__(self, file_path: str) -> torch.Tensor:
        """
        Processes an audio file into a mel spectrogram tensor.

        Args:
            file_path (str): Path to the audio file.

        Returns:
            torch.Tensor: Mel spectrogram tensor.
        """
        y, sr = load_audio(file_path, sample_rate=self.sample_rate)
        mel_spec = mel_spectrogram(y, sr)
        
        mel_spec_tensor = torch.tensor(mel_spec[np.newaxis, np.newaxis, ...], dtype=torch.float32)
        mel_spec_tensor = mel_spec_tensor.squeeze(0)
        mel_spec_tensor = mel_spec_tensor.unsqueeze(0)
        mel_spec_tensor = F.interpolate(mel_spec_tensor, size=self.size)
        
        return mel_spec_tensor
