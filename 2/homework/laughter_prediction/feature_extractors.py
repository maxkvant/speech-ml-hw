import os
import tempfile
import librosa

import pandas as pd
import numpy as np


class FeatureExtractor:
    def extract_features(self, wav_path):
        """
        Extracts features for classification ny frames for .wav file

        :param wav_path: string, path to .wav file
        :return: pandas.DataFrame with features of shape (n_chunks, n_features)
        """
        raise NotImplementedError("Should have implemented this")


class PyAAExtractor(FeatureExtractor):
    """Python Audio Analysis features extractor"""
    def __init__(self):
        self.extract_script = "./extract_pyAA_features.py"
        self.py_env_name = "ipykernel_py2"

    def extract_features(self, wav_path):
        with tempfile.NamedTemporaryFile() as tmp_file:
            feature_save_path = tmp_file.name
            cmd = "python \"{}\" --wav_path=\"{}\" " \
                  "--feature_save_path=\"{}\"".format(self.extract_script, wav_path, feature_save_path)
            os.system("source activate {}; {}".format(self.py_env_name, cmd))

            feature_df = pd.read_csv(feature_save_path)
        return feature_df


class LibrosaExtractor(FeatureExtractor):
    """Python Audio Analysis features extractor"""
    def __init__(self, window_size=0.01):
        self.window_size = window_size

    def extract_features(self, wav_path):
        y, sr = librosa.load(wav_path)

        frame_len = int(sr * self.window_size)
        mel = []
        mfcc = []

        for frame_start in range(0, len(y) - frame_len, frame_len):
            frame = y[frame_start: frame_start + frame_len]

            mel_feature = librosa.feature.melspectrogram(frame, sr)
            mel_feature = librosa.power_to_db(mel_feature)
            mel.append(np.average(mel_feature, axis=1))

            mfcc_feature = librosa.feature.mfcc(frame, sr)
            mfcc.append(np.average(mfcc_feature, axis=1))

        mel = np.asarray(mel)
        mfcc = np.asarray(mfcc)

        mel_df  = pd.DataFrame(mel,  columns=[ "mel_{0:03d}".format(i) for i in range( mel.shape[1])])
        mfcc_df = pd.DataFrame(mfcc, columns=["mfcc_{0:03d}".format(i) for i in range(mfcc.shape[1])])

        feature_df = pd.concat([mel_df, mfcc_df], axis=1)

        return feature_df