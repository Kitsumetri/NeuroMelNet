import torch
import torch.nn as nn
import librosa as lb
import numpy as np

import constants as consts

def load_checkpoint(load_path: str,  
                    model: nn.Module, 
                    pretrained_eval=False, 
                    optimizer=None):
    checkpoint = torch.load(load_path)
    
    if pretrained_eval:
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    if optimizer is None:
        raise RuntimeError('Optimizer cannot be None')

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

    return model, optimizer, epoch

def load_audio(file_path, sample_rate):
    audio, sr = lb.load(file_path, sr=sample_rate)
    return audio, sr


def mel_spectrogram(audio, sample_rate, 
                    n_fft=consts.N_FFT, 
                    hop_length=consts.HOP_LENGTH,
                    n_mels=consts.N_MELS):
    
    mel_spec = lb.feature.melspectrogram(y=audio, 
                                         sr=sample_rate,    
                                        n_fft=n_fft,
                                        hop_length=hop_length, 
                                        n_mels=n_mels)
    
    mel_spec_db = lb.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db