import numpy as np
import librosa
import matplotlib.pyplot as plt
import argparse
import os
import random
import yaml
import soundfile as sf

eps = 1e-9
sample_rate = 16000


class NoiseAdder:
    def __init__(self, fragment_generators, random_start=True):
        self.fragment_generators = fragment_generators
        self.random_start = random_start

    def add_noise(self, sound):
        n = len(sound)
        assert n > 1

        noise = np.zeros(n)
        for fragment_gen in self.fragment_generators:
            fragment = fragment_gen()
            rand_shift = random.randint(0, len(fragment))
            fragment = np.roll(fragment, rand_shift if self.random_start else 0)

            noise += np.tile(fragment, n // len(fragment) + 1)[:n]
        return sound + noise


def load_audio(file_path):
    data, _ = librosa.core.load(file_path, sample_rate)
    return data


def audio_getter(directory, ratio):
    filenames = [directory + os.sep + filename for filename in os.listdir(directory)]
    filenames = [filename for filename in filenames
                 if os.path.isfile(filename) and (filename.endswith(".wav") or filename.endswith(".flac"))]

    assert len(filenames) > 0
    def gen_audio():
        filename = random.choice(filenames)
        return ratio * load_audio(filename)
    return gen_audio


def create_noise_adder(noise_yaml):
    fragment_generators = []
    for item in noise_yaml["noise_info"]:
        path = item["path"]
        ratio = float(item["ratio"])
        if os.path.isfile(path) and os.path.exists(path):
            fragment_generators.append(lambda: load_audio(path) * ratio)
        elif os.path.isdir(path) and os.path.exists(path):
            fragment_generators.append(audio_getter(path, ratio))
        else:
            raise("no file or directory " + path)
    return NoiseAdder(fragment_generators)


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help="Input file (.wav or .flac)")
    parser.add_argument('-n', '--noise', help="Noise discription (.yaml)", default='noise.yaml')
    parser.add_argument('-o', '--output', help="Output file (.wav or .flac)", default='output.wav')
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    input_file = args.input
    output_file = args.output
    with open(args.noise, 'r') as f:
        noise_yaml = yaml.load(f)

    assert input_file.endswith(".wav") or input_file.endswith(".flac")
    assert output_file.endswith(".wav") or output_file.endswith(".flac")

    sound = load_audio(input_file)
    output = create_noise_adder(noise_yaml).add_noise(sound)

    if output_file.endswith(".wav"):
        librosa.output.write_wav(output_file, output, sample_rate)
    elif output_file.endswith(".flac"):
        sf.write(output_file, output, sample_rate, format='flac', subtype='PCM_24')


if __name__ == '__main__':
    main()

