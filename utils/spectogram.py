"""
reference:http://www.frank-zalkow.de/en/code-snippets/create-audio-spectrograms-with-python.html?i=1
"""
import librosa
from os import listdir
from os.path import isfile, join
import numpy as np
from matplotlib import pyplot as plt
from numpy.lib import stride_tricks
from . import metadata


def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))

    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(int(np.floor(frameSize / 2.0))), sig)
    # cols for windowing
    cols = np.ceil((len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))

    frames = stride_tricks.as_strided(samples, shape=(int(cols), frameSize),
                                      strides=(samples.strides[0] * hopSize, samples.strides[0])).copy()
    frames *= win

    return np.fft.rfft(frames)


def logscale_spec(spec, sr=44100, factor=20.):
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins - 1) / max(scale)
    scale = np.unique(np.round(scale))

    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):
        if i == len(scale) - 1:
            newspec[:, i] = np.sum(spec[:, int(scale[i]):], axis=1)
        else:
            newspec[:, i] = np.sum(spec[:, int(scale[i]):int(scale[i + 1])], axis=1)

    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins * 2, 1. / sr)[:freqbins + 1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale) - 1:
            freqs += [np.mean(allfreqs[int(scale[i]):])]
        else:
            freqs += [np.mean(allfreqs[int(scale[i]):int(scale[i + 1])])]

    return newspec, freqs


def plotstft(audiopath, filepath, binsize=2 ** 8, colormap="jet"):
    samples, samplerate = librosa.load(audiopath, sr=44100)
    s = stft(samples, binsize)

    sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)

    ims = 20. * np.log10(np.abs(sshow) / 10e-6)  # amplitude to decibel

    timebins, freqbins = np.shape(ims)

    print("timebins: ", timebins)
    print("freqbins: ", freqbins)

    plt.figure(figsize=(15, 7.5))
    plt.imshow(np.transpose(ims), origin="lower", aspect="auto", interpolation="none")
    plt.xlim([0, timebins - 1])
    plt.axis("off")
    plt.savefig(filepath, bbox_inches="tight", pad_inches=0)
    plt.clf()


def dir_to_spectograms(audio_dir, spectogram_dir):
    """ Create Spectogram of all .wav files in a directory.

        :param audio_dir: path of directory with audio files
        :param spectogram_dir: path to save spectograms
        """
    file_names = [file for file in listdir(audio_dir) if isfile(join(audio_dir, file)) and '.wav' in file]

    for file_name in file_names:
        print("Generating Spectogram Visual for Audio " + file_name)
        audio_path = audio_dir + file_name
        spectogram_path = spectogram_dir + file_name.replace('.wav', '.png')
        plotstft(audio_path, spectogram_path)
        print("Spectogram saved!!!")

    print("All the audio files have been converted to Spectogram present at " + spectogram_dir)


if __name__ == '__main__':
    audio_dir = '/Users/Anuj Shah/A-dataset-of-Indian-Folk-Songs/audio chunks/'
    spectogram_dir = '/Users/Anuj Shah/A-dataset-of-Indian-Folk-Songs/features/spectograms/ '
    for languages in metadata.keys():
        dir_to_spectograms(audio_dir + languages+'/', spectogram_dir + languages+'/')
