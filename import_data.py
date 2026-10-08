import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Audio, display
import os



def load_data():
    # Download the LibriSpeech dataset
    librispeech_train = torchaudio.datasets.LIBRISPEECH(".", url="train-clean-100",download=True)
    librispeech_test = torchaudio.datasets.LIBRISPEECH(".", url="test-clean", download=True)
    return librispeech_train, librispeech_test

def load_loaders(librispeech_train, librispeech_test):
    # Create DataLoaders for the training and testing datasets
    train_loader = torch.utils.data.DataLoader(librispeech_train, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(librispeech_test, batch_size=32, shuffle=True)
    return train_loader, test_loader

def show_waveform(speech_train):
    # Plot example waveform and display audio
    example = speech_train[0]
    print(f"Waveform shape: {example[0].shape}")
    plt.figure()
    plt.plot(example[0].t().numpy())
    plt.xlabel("Time (Hz)")
    plt.ylabel("Amplitude")
    plt.show()

    sample_rate = example[1]
    display(Audio(example[0].numpy(), rate=sample_rate))



