import torch
import torchaudio

def load_data():
    ljspeech = torchaudio.datasets.LJSPEECH(".", download=True)
    data_loader = torch.utils.data.DataLoader(ljspeech, batch_size=32, shuffle=True)
    return data_loader




