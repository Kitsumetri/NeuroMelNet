import torch


DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

SAMPLE_RATE = 10_000
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 256
AUDIO_SIZE = (128, 128)
