from __future__ import division, print_function
from os import listdir
from os.path import isfile, join
import librosa
import librosa.feature as lf
import numpy as np
import pandas as pd
from . import metadata

allowed_strategy = ['mean', 'median', 'stddev']


def calc_features(audio_path):
    """ Calculate 19 features for a audio file.
    These Features includes [chroma_stft, rmse, spectral_centroid, spectral_bandwidth, spectral_rolloff, zero_crossing_rate, mfcc 1 = mfcc 13]

    :param audio_path: path of the wav file
    """
    features = []
    sample, sample_rate = librosa.load(audio_path)
    chroma_stft = lf.chroma_stft(sample, sample_rate)
    features.append(chroma_stft)
    rmse = lf.rmse(sample)
    features.append(rmse)
    spectral_centroid = lf.spectral_centroid(sample, sample_rate)
    features.append(spectral_centroid)
    spectral_bandwidth = lf.spectral_bandwidth(sample, sample_rate)
    features.append(spectral_bandwidth)
    spectral_rolloff = lf.spectral_rolloff(sample, sample_rate)
    features.append(spectral_rolloff)
    zero_crossing_rate = lf.zero_crossing_rate(sample)
    features.append(zero_crossing_rate)
    mfcc_features = lf.mfcc(sample, sample_rate, n_mfcc=13)
    for mfcc_feature in mfcc_features:
        features.append(mfcc_feature)

    return features


def convert_into_vector(audio_path, strategy):
    """

    :param audio_path:  path of the wav file
    :param strategy: Can be mean, median and stddev.
    """
    get_features = calc_features(audio_path)
    feature_vector = []
    if strategy == 'mean':
        for feature in get_features:
            feature_vector.append(np.mean(feature))
    elif strategy == 'median':
        for feature in get_features:
            feature_vector.append(np.median(feature))
    elif strategy == 'stddev':
        for feature in get_features:
            feature_vector.append(np.std(feature))

    return feature_vector


def get_category_feature(audio_dir, strategy, label):
    """ Create MelSpectogram of a .wav file

        :param audio_path: path of the wav file
        :param strategy: Can be mean, median and stddev.
        :param label: Categorise the output label.
    """

    file_names = [file for file in listdir(audio_dir) if isfile(join(audio_dir, file)) and '.wav' in file]

    audio_category_feature_vector = []
    for file_name in file_names:
        feature_vector = convert_into_vector(audio_dir, strategy)
        feature_vector.append(label)
        audio_category_feature_vector.append(feature_vector)

    return audio_category_feature_vector


if __name__ == '__main__':
    audio_dir = '/Users/Anuj Shah/A-dataset-of-Indian-Folk-Songs/audio chunks/'
    save_path = '/Users/Anuj Shah/A-dataset-of-Indian-Folk-Songs/'
    for strategy in allowed_strategy:
        label = 0
        features_all_category = []
        for languages in metadata.keys():
            features_all_category.extend(get_category_feature(audio_dir + languages + '/', strategy, label))
            label += 1
        features_all_category = pd.DataFrame(features_all_category)
        features_all_category.to_csv(strategy+'_features.csv', index=False)

