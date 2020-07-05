from __future__ import division, print_function
from os import listdir
from os.path import isfile, join
import librosa
import librosa.display
import librosa.feature
import matplotlib.pyplot as plt
from . import metadata


def wav_to_mfccvisual(audio_path, save_path):
    """ Create MFCC Visualisation of a .wav file

    :param audio_path: path of the wav file
    :param save_path: path where the generated MFCC Visualisation is to  be saved.
    """
    sample, sample_rate = librosa.load(audio_path)
    mfcc_features = librosa.feature.mfcc(sample, sample_rate, n_mfcc=13)
    plt.figure(figsize=(15, 7.5))
    librosa.display.specshow(mfcc_features)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)


def dir_to_mfccvisual(audio_dir, mfcc_visual_dir):
    """ Create MFCC Visualisation of all .wav files in a directory.

    :param audio_dir: path of directory with audio files
    :param mfcc_visual_dir: path to save MFCC Visualisations
    """
    file_names = [file for file in listdir(audio_dir) if isfile(join(audio_dir, file)) and '.wav' in file]

    for file_name in file_names:
        print("Generating MFCC Visual for Audio " + file_name)
        audio_path = audio_dir + file_name
        mfcc_visual_path = mfcc_visual_dir + file_name.replace('.wav', '.png')
        wav_to_mfccvisual(audio_path, mfcc_visual_path)
        print("MFCC Visual saved!!!")

    print("All the audio files have been converted to MFCC Visualisations present at " + mfcc_visual_dir)


if __name__ == '__main__':
    audio_dir = '/Users/Anuj Shah/A-dataset-of-Indian-Folk-Songs/audio chunks/'
    mfccvisual_dir = '/Users/Anuj Shah/A-dataset-of-Indian-Folk-Songs/features/mfcc_visual/ '
    for languages in metadata.keys():
        dir_to_mfccvisual(audio_dir + languages+'/', mfccvisual_dir + languages+'/')

