import numpy as np
import random
import librosa
import matplotlib.pyplot as plt
import argparse
import soundfile as sf


eps = 1e-9
samplerate = 16000


class NoiseAdder:
    def __init__(self, fragment, ratio=0.1, normalize=True, random_start=True):
        assert len(fragment) > 1
        self.fragment = fragment.copy()
        self.normalize = normalize
        self.ratio = ratio
        self.random_start = random_start

    def add_noise(self, sound):
        n = len(sound)
        assert n > 1

        rand_shift = random.randint(0, len(self.fragment))
        fragment = np.roll(self.fragment, rand_shift if self.random_start else 0)

        noise = np.tile(fragment, n // len(fragment) + 1)[:n]
        if self.normalize:
            noise = noise * np.std(sound) / np.std(noise)

        return sound + self.ratio * noise


def load_audio(file_path):
    data, _ = librosa.core.load(file_path, samplerate)
    return data

def plot_time_series(data):
    fig = plt.figure(figsize=(14, 8))
    plt.title('Raw wave ')
    plt.ylabel('Amplitude')
    plt.plot(np.linspace(0, 1, len(data)), data)
    plt.show()


def createparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help="Input file (.wav or .flac)", default='speech.wav')
    parser.add_argument('-n', '--noise', help="Noise file (.wav or .flac)", default='office01.wav')
    parser.add_argument('-r', '--ratio', help="Noise ratio", default=0.5)
    parser.add_argument('-a', '--normalize', help="Normalize", default=True)
    parser.add_argument('-o', '--output', help="Output file (.wav or .flac)", default='output.wav')
    return parser


def main():
    parser = createparser()
    args = parser.parse_args()
    input_file = args.input
    noise_file = args.noise
    ratio = args.ratio
    normalize = args.normalize
    output_file = args.output

    sound = load_audio(input_file)
    noise = load_audio(noise_file)

    sound = np.asarray(sound)
    noise = np.asarray(noise)

    output = NoiseAdder(noise, ratio=ratio, normalize=normalize).add_noise(sound)

    librosa.output.write_wav(output_file, output, samplerate)


if __name__ == '__main__':
    main()

