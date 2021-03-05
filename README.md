# HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis

A Pytorch1.4 implementation of HiFi-GAN vocoder adapted from the official github repository, with slight modification on Mel-spectrogram preprocessing.

- [Paper](https://arxiv.org/abs/2010.05646)
- [Official Github Repo](https://github.com/jik876/hifi-gan)
- [Official Demo Site](https://jik876.github.io/hifi-gan-demo)


In the original repo, input mel-spectrogram of target audio is extracted base on **pytorch.stft** on the fly, while in this repo, input mel-spectrogram of target audio is loaded from stored result of data preprocess.

## Training

1. Prepare your dataset.
    - Extract mel-spectrogram from training audio data;
    - Divide dataset into training set & validation set;
    - (Optional) Implement your own `get_dataset_filelist` method in `meldataset.py` to handle metadata. 
2. Specify following parameters and run `train.py` to start trainging:
    - `input_training_file`: metadata file of training set;
    - `input_validation_file`: metadata file of validation set;
    - `input_mel_dir`: directory where input mel-spectrograms are stored;
    - `input_wavs_dir`: directory where target audio data are stored (in `.npy`)
    - `checkpoint_path`: directory to save model & trainging logs
    - `config`: path of configuration file (in JSON, e.g. `config.json`)
    - other optional parameters.

## Inference

1. Download a **pretrained HiFi-GAN generator model** and corresponding **JSON configuration file** into **the same directory**.
2. Follow the example in `inference.py`, construct a `Vocoder` instance **by specifying the path of the checkpoint file of Generator** to be loaded.
3. Waveform now could be generated from input mel-spectrogram with interface `Vocoder.mel2wav`.
