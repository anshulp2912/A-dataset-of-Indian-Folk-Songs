from __future__ import division, print_function
from os import listdir
from os.path import isfile, join
import librosa
import librosa.display
import librosa.feature
import matplotlib.pyplot as plt
import numpy as np
from . import metadata


def wav_to_melspectrogram(audio_path, save_path):
    """ Create MelSpectogram of a .wav file

    :param audio_path: path of the wav file
    :param save_path: path where the generated MelSpectogram is to  be saved.
    """
    sample, sample_rate = librosa.load(audio_path)
    mel_features = librosa.feature.melspectrogram(sample, sample_rate)
    plt.figure(figsize=(15, 7.5))
    S_dB = librosa.power_to_db(mel_features, ref=np.max)
    librosa.display.specshow(S_dB, sr=sample_rate)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)


def dir_to_melspectrogram(audio_dir, melspectrogram_dir):
    """ Create MelSpectogram of all .wav files in a directory.

    :param audio_dir: path of directory with audio files
    :param melspectogram_dir: path to save MelSpectograms
    """
    file_names = [file for file in listdir(audio_dir) if isfile(join(audio_dir, file)) and '.wav' in file]

    for file_name in file_names:
        print("Generating MelSpectrogram for Audio " + file_name)
        audio_path = audio_dir + file_name
        melspectrogram_path = melspectrogram_dir + file_name.replace('.wav', '.png')
        wav_to_melspectrogram(audio_path, melspectrogram_path)
        print("MelSpectrogram saved!!!")

    print("All the audio files have been converted to MelSpectograms present at "+melspectrogram_dir)


if __name__ == '__main__':
    audio_dir = '/Users/Anuj Shah/A-dataset-of-Indian-Folk-Songs/audio chunks/'
    melspectrogram_dir = '/Users/Anuj Shah/A-dataset-of-Indian-Folk-Songs/features/melspectrograms/ '
    for languages in metadata.keys():
        dir_to_melspectrogram(audio_dir + languages+'/', melspectrogram_dir + languages+'/')