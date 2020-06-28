import matplotlib
import matplotlib.pylab as plt

import IPython.display as ipd

import sys
import os
#sys.path.append('waveglow/')
import numpy as np
import torch
from scipy.io.wavfile import write
from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
from denoiser import Denoiser

MAX_WAV_VALUE = 32768.0

def plot_data(data, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='bottom', 
                       interpolation='none')
    plt.savefig("result.png")
					   
hparams = create_hparams()
hparams.sampling_rate = 22050

checkpoint_path = "outdir/checkpoint_61000.pt"
model = load_model(hparams)
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
_ = model.cuda().eval().half()

waveglow_path = 'waveglow/checkpoints/waveglow_2000_update.pt'
waveglow = torch.load(waveglow_path)['model']
waveglow.cuda().eval().half()
for k in waveglow.convinv:
    k.float()
denoiser = Denoiser(waveglow)

text = "na2 u7 hit4 lo7 bi2 kok4 si5 kan1 leh4 pinn3 hiah4 e5 uann2 ko1 tshut4 thau5"
sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
	
mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
plot_data((mel_outputs.float().data.cpu().numpy()[0],
           mel_outputs_postnet.float().data.cpu().numpy()[0],
           alignments.float().data.cpu().numpy()[0].T))
		   
file_name = 'output_demo_hw3_2_10'
with torch.no_grad():
    audio = waveglow.infer(mel_outputs_postnet, sigma=1)
    audio = audio * MAX_WAV_VALUE
audio = audio.squeeze()
audio = audio.cpu().numpy()
audio = audio.astype('int16')
audio_path = os.path.join("{}_synthesis.wav".format(file_name))
write(audio_path, hparams.sampling_rate, audio)
print(audio_path)


file_name = 'output_denoiser_demo_hw3_2_10'
with torch.no_grad():
    audio = waveglow.infer(mel_outputs_postnet, sigma=1)
    audio = audio * MAX_WAV_VALUE
audio_denoised = denoiser(audio, strength=0.01)
audio_denoised = audio_denoised.squeeze()
audio_denoised = audio_denoised.cpu().numpy()
audio_denoised = audio_denoised.astype('int16')
audio_path = os.path.join("{}_synthesis.wav".format(file_name))
write(audio_path, hparams.sampling_rate, audio_denoised)
print(audio_path)

