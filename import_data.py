import torch
import torchaudio
import matplotlib.pyplot as plt

def load_data():
    librispeech_train = torchaudio.datasets.LIBRISPEECH(".", url="train-clean-100",download=True)
    librispeech_test = torchaudio.datasets.LIBRISPEECH(".", url="test-clean", download=True)
    return librispeech_train, librispeech_test

def load_loaders(librispeech_train, librispeech_test):
    train_loader = torch.utils.data.DataLoader(librispeech_train, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(librispeech_test, batch_size=32, shuffle=True)
    return train_loader, test_loader

def show_waveform(train_loader):
    example = train_loader.__getitem__(0)
    print(f"Waveform shape: {example[0].shape}")
    plt.figure()
    plt.plot(example[0].t().numpy())
    plt.show()



