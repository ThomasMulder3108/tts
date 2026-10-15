import torch
import torchaudio
from torchaudio.pipelines import WAV2VEC2_BASE, WAV2VEC2_ASR_BASE_960H
from import_data import load_data
import matplotlib.pyplot as plt
from IPython.display import Audio, display
import soundfile as sf
import os
from pydub import AudioSegment
import numpy as np
import whisper

def load_model():
    # Load the model and tokenizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bundle = WAV2VEC2_ASR_BASE_960H
    print(f"Sample rate: {bundle.sample_rate}")
    #print(f"Labels: {bundle.get_labels()}")

    model = bundle.get_model().to(device)

    return model, bundle, device

def feature_extraction(model, waveform, device, verbose=False):
    with torch.inference_mode():
        features, _ = model.extract_features(waveform.to(device))

    if verbose:
        fig, ax = plt.subplots(len(features), 1, figsize=(16, 4.3 * len(features)))
        for i, feature in enumerate(features):
            ax[i].imshow(feature[0].cpu(), interpolation='nearest')
            ax[i].set_title(f'Feature {i+1}')
            ax[i].set_xlabel('Feature dimension')
            ax[i].set_ylabel('Time step')
        fig.tight_layout()

    return features

def classification(model, waveform, device, bundle, verbose=False):
    with torch.inference_mode():
        emissions, _ = model(waveform.to(device))

    if verbose:
        plt.imshow(emissions[0].cpu().T, interpolation='nearest')
        plt.title('Classification Emissions')
        plt.xlabel('Time step')
        plt.ylabel('Class')
        plt.tight_layout()
        print(f"Labels: {bundle.get_labels()}")

    return emissions

def main_asr(audio_file=None, verbose=False):
    print(audio_file)

    # Load model and tokenizer
    model, bundle, device = load_model()

    # Load data
    train, test = load_data()

    if audio_file is None:
        # Use the first sample from the test dataset
        waveform, sample_rate, transcript, speaker_ID, Chapter_ID, Utterance_ID = test[0]

    else: 
        # Load the audio file
        path = os.path.join(audio_file)

        with open(path, 'rb') as f:
            audio = AudioSegment.from_file(f)
            sample_rate = audio.frame_rate
            waveform = audio.get_array_of_samples()
            waveform = np.asarray(waveform)
            waveform = torch.from_numpy(waveform)
            # Turn dtype into float32 
            waveform = waveform.float()

        # Reshape waveform to have shape (1, num_samples) if it's a 1D tensor
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

    # Resample if necessary
    if sample_rate != bundle.sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=bundle.sample_rate)
        waveform = resampler(waveform)

    # Feature extraction
    features = feature_extraction(model, waveform, device, verbose)

    # Classification
    emissions = classification(model, waveform, device, bundle, verbose)

    decoder = GreedyCTCDecoder(bundle.get_labels())
    transcription = decoder(emissions[0])

    # Output
    print(f"Transcription: {transcription}")
    display(Audio(waveform.cpu().numpy(), rate=bundle.sample_rate))

    transcription = transcription.replace("|", " ").lower()

    return transcription


class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emissions):
        # Get the most likely token at each time step
        indices = torch.argmax(emissions, dim=-1)
        # Convert tokens to characters
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        transcription = ''.join([self.labels[i] for i in indices])
        return transcription


def whisper_model(audio_file=None):
    _, test = load_data()

    if audio_file is None:
        # Use the first sample from the test dataset
        audio_file = test[0][0]  # Assuming the first element is the waveform

    model = whisper.load_model("small")
    result = model.transcribe(audio_file)
    return result["text"]
    




