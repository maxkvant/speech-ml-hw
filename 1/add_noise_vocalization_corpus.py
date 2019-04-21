from add_noise import create_noise_adder, load_audio, sample_rate
import librosa
import soundfile as sf
import os
import time
import yaml


def output_audio(output, output_file):
    if output_file.endswith(".wav"):
        librosa.output.write_wav(output_file, output, sample_rate)
    elif output_file.endswith(".flac"):
        sf.write(output_file, output, sample_rate, format='flac', subtype='PCM_24')


def main():
    prefixes = ["S27", "S28", "S29"]
    print([prefix + "*.wav" for prefix in prefixes])
    def check(filename):
        for prefix in prefixes:
            if filename.startswith(prefix):
                return True
        return False

    dir_from = "vocalizationcorpus/data/"
    dir_to = "vocalizationcorpus_noisy_2/data/"
    with open('noise-vocalizationcorpus.yaml', 'r') as f:
        noise_yaml = yaml.load(f)
    print(noise_yaml)
    noise_adder = create_noise_adder(noise_yaml)

    files = [file for file in os.listdir(dir_from) if check(file)]
    files.sort()

    start_time = time.time()

    for iter, filename in enumerate(files, start=1):
        audio = load_audio(dir_from + filename)
        file_to = dir_to + filename
        output = noise_adder.add_noise(audio)
        output_audio(output, file_to)

        if iter & (iter - 1) == 0:
            print("iter {}/{} {} {}s".format(iter, len(files), filename, time.time() - start_time))


if __name__ == '__main__':
    main()
