import os
import argparse
import gradio as gr
from timeit import default_timer as timer
import torch
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from model.bart import BartCaptionModel
from utils.audio_utils import load_audio, STR_CH_FIRST

if os.path.isfile("transfer.pth") == False:
    torch.hub.download_url_to_file('https://huggingface.co/seungheondoh/lp-music-caps/resolve/main/transfer.pth', 'transfer.pth')
    torch.hub.download_url_to_file('https://huggingface.co/seungheondoh/lp-music-caps/resolve/main/electronic.mp3', 'electronic.mp3')
    torch.hub.download_url_to_file('https://huggingface.co/seungheondoh/lp-music-caps/resolve/main/orchestra.wav', 'orchestra.wav')

device = "cuda:0" if torch.cuda.is_available() else "cpu"

example_list = ['electronic.mp3', 'orchestra.wav']
model = BartCaptionModel(max_length = 128)
pretrained_object = torch.load('./transfer.pth', map_location='cpu')
state_dict = pretrained_object['state_dict']
model.load_state_dict(state_dict)
if torch.cuda.is_available():
    torch.cuda.set_device(device)
model = model.cuda(device)
model.eval()

def get_audio(audio_path, duration=10, target_sr=16000):
    n_samples = int(duration * target_sr)
    audio, sr = load_audio(
        path= audio_path,
        ch_format= STR_CH_FIRST,
        sample_rate= target_sr,
        downmix_to_mono= True,
    )
    if len(audio.shape) == 2:
        audio = audio.mean(0, False)  # to mono
    input_size = int(n_samples)
    if audio.shape[-1] < input_size:  # pad sequence
        pad = np.zeros(input_size)
        pad[: audio.shape[-1]] = audio
        audio = pad
    ceil = int(audio.shape[-1] // n_samples)
    audio = torch.from_numpy(np.stack(np.split(audio[:ceil * n_samples], ceil)).astype('float32'))
    return audio

def captioning(audio_path):
    audio_tensor = get_audio(audio_path = audio_path)
    if device is not None:
        audio_tensor = audio_tensor.to(device)
    with torch.no_grad():
        output = model.generate(
            samples=audio_tensor,
            num_beams=5,
        )
    inference = ""
    number_of_chunks = range(audio_tensor.shape[0])
    for chunk, text in zip(number_of_chunks, output):
        time = f"[{chunk * 10}:00-{(chunk + 1) * 10}:00]"
        inference += f"{time}\n{text} \n \n"
    return inference

title = "Interactive demo: Music Captioning 🤖🎵"
description = “”
article = "<p style='text-align: center'><a href='https://github.com/seungheondoh/lp-music-caps' target='_blank'>LP-MusicCaps Github</a> | <a href='#' target='_blank'>LP-MusicCaps Paper</a></p>"


files_in_directory = os.listdir()
audio_files = [file for file in files_in_directory if file.endswith('.mp3') or file.endswith('.wav')]

from pandas import DataFrame
df = pd.DataFrame(columns=["file_name", "captioning"])


for filename in audio_files:
    df.loc[len(df)] = [filename, captioning(f"/kaggle/working/lp-music-caps/demo/Kaggle_music/{filename}")]
    print(f"{filename} Done")

file_path = 'music_captioned.csv'
df.to_csv(file_path, index=False)

