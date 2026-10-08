import torch
import torchaudio
from torchaudio.pipelines import WAV2VEC2_BASE, WAV2VEC2_ASR_BASE_960H
from import_data import load_data
import matplotlib.pyplot as plt
from IPython.display import Audio, display

def load_model():
    bundle = WAV2VEC2_BASE
    model = bundle.get_model()
    return model

def load_tokenizer():
    bundle = WAV2VEC2_ASR_BASE_960H
    tokenizer = bundle.get_labels()
    return tokenizer

def transcribe_audio(model, tokenizer, audio):
    with torch.no_grad():
        emissions, _ = model(audio)
    tokens = torch.argmax(emissions, dim=-1)
    transcription = ''.join([tokenizer[token] for token in tokens[0]])
    return transcription

def main(audio_file=None, verbose=False):
    # Load the model and tokenizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bundle = WAV2VEC2_ASR_BASE_960H
    print(f"Sample rate: {bundle.sample_rate}")
    #print(f"Labels: {bundle.get_labels()}")

    model = bundle.get_model().to(device)

    train, test = load_data()

    if audio_file is None:
        # Use the first sample from the training dataset
        waveform, sample_rate, _, _, _, _ = train[0]

    else: 
        # Load the audio file
        waveform, sample_rate = torchaudio.load(audio_file)

    # Resample if necessary
    if sample_rate != bundle.sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=bundle.sample_rate)
        waveform = resampler(waveform)

    # Feature extraction
    with torch.inference_mode():
        features, _ = model.extract_features(waveform.to(device))

    if verbose:
        fig, ax = plt.subplots()
        for i, feature in enumerate(features):
            ax.plot(feature[0].cpu().numpy(), label=f'Layer {i}')
            ax.set_title(f'Features from Layer {i}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Feature Value')
        fig.tight_layout()

    # Classification
    with torch.inference_mode():
        emissions, _ = model(waveform.to(device))

    if verbose:
        fig, ax = plt.subplots()
        ax.plot(emissions[0].cpu().numpy())
        ax.set_title('Emissions')
        ax.set_xlabel('Time')
        ax.set_ylabel('Emission Value')
        fig.tight_layout()

    decoder = GreedyCTCDecoder(bundle.get_labels())
    transcription = decoder(emissions[0])

    print(f"Transcription: {transcription}")
    display(Audio(waveform.cpu().numpy(), rate=bundle.sample_rate))



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


    




