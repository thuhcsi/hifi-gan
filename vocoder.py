from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import glob
import json
import os
from pathlib import Path

import torch
from scipy.io.wavfile import write
from tqdm import tqdm

from env import AttrDict
from mel_extractor.mel import wav2mel
from meldataset import MAX_WAV_VALUE
from models import Generator


class Vocoder:
    def __init__(self, checkpoint_dir, checkpoint_file, config_file="config.json", device="cpu"):
        checkpoint_dir = Path(checkpoint_dir)
        self.config = self.load_config(checkpoint_dir / config_file)
        self.generator = self.load_generator(checkpoint_dir / checkpoint_file, self.config, device)


    def load_config(self, config_file):
        with open(config_file) as f:
            data = f.read()
            json_config = json.loads(data)
        return AttrDict(json_config)

    def load_generator(self, checkpoint_file, config, device):
        torch.manual_seed(config.seed)
        if device=="cuda" and torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        print("Loading '{}'".format(checkpoint_file))
        assert checkpoint_file.is_file()
        generator = Generator(config).to(device)
        checkpoint_dict = torch.load(checkpoint_file, map_location=device)
        generator.load_state_dict(checkpoint_dict['generator'])
        generator.eval()
        generator.remove_weight_norm()
        print("Complete.")

        return generator
    
    def mel2wav(self, mel, output_file=None):
        with torch.no_grad():
            x = torch.FloatTensor(mel.T).unsqueeze(0)
            y_g_hat = self.generator(x)
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')

        if output_file:
            write(output_file, self.config.sampling_rate, audio)
        return audio


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_wavs_dir', default='emodb6_test_files')
    parser.add_argument('-o', '--output_dir', default='generated_files')
    parser.add_argument('--checkpoint_file', default='emodb6/g_00800000')
    args = parser.parse_args()

    input_wavs_dir = Path(args.input_wavs_dir)
    output_wavs_dir = Path(args.output_dir)
    output_wavs_dir.mkdir(exist_ok=True)


    synthesizer = Vocoder("emodb6_hw", "g_00700000", device="cpu")

    mels = [(wav2mel(wavfile)[0], wavfile.name) for wavfile in input_wavs_dir.iterdir()]
    for mel, savename in tqdm(mels):
        synthesizer.mel2wav(mel, output_file=str(output_wavs_dir/savename))


if __name__ == '__main__':
    main()

