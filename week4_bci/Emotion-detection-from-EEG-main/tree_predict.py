# Copyright (c) 2020 Shubh Pachchigar
"""
EEG Data is taken from DEAP
The training data was taken from DEAP.
See my:
- Github profile: https://github.com/shubhe25p
- Email: shubhpachchigar@gmail.com
"""

import csv
import numpy as np
import scipy.io as sio
import cv2
from sklearn.tree import DecisionTreeClassifier  # Decision Tree 추가
from sklearn.preprocessing import StandardScaler

sampling_rate = 128
number_of_channel = 32  # considering only head electrodes
eeg_in_second = 63  # length of each trial
number_of_eeg = sampling_rate * eeg_in_second  # total inputs from a single channel

channel_names = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz',
                 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2']


class PredictEmotion:
    """
    Receives EEG data preprocessing and predicts emotion using Decision Tree.
    """

    def __init__(self):
        """
        Initializes training data and trains Decision Tree classifiers for arousal and valence.
        """
        # Load training data
        self.train_features = self.get_csv("train_std.csv")
        self.class_arousal = self.get_csv("class_arousal.csv").ravel()  # 1D array로 변환
        self.class_valence = self.get_csv("class_valence.csv").ravel()

        # 데이터 스케일링 (정규화)
        self.scaler = StandardScaler()
        self.train_features = self.scaler.fit_transform(self.train_features)

        # Initialize Decision Tree classifiers
        self.tree_arousal = DecisionTreeClassifier(criterion="entropy", max_depth=10, random_state=42)
        self.tree_valence = DecisionTreeClassifier(criterion="entropy", max_depth=10, random_state=42)

        # Train the models
        self.tree_arousal.fit(self.train_features, self.class_arousal)
        self.tree_valence.fit(self.train_features, self.class_valence)

    def get_csv(self, path):
        """
        Load CSV file and convert to numpy array.
        """
        data_csv = np.loadtxt(path, delimiter=",", dtype=np.float64)
        return data_csv

    def get_feature(self, all_channel_data):
        """
        Extracts features (standard deviation) from EEG frequency bands.
        """
        delta, theta, alpha, beta, gamma = self.get_frequency(all_channel_data)

        delta_std = np.std(delta, axis=1)
        theta_std = np.std(theta, axis=1)
        alpha_std = np.std(alpha, axis=1)
        beta_std = np.std(beta, axis=1)
        gamma_std = np.std(gamma, axis=1)

        feature = np.concatenate([delta_std, theta_std, alpha_std, beta_std, gamma_std])
        return feature.reshape(1, -1)  # sklearn 입력 형태로 변환

    def get_frequency(self, all_channel_data):
        """
        Extracts EEG frequency bands: Delta, Theta, Alpha, Beta, Gamma.
        """
        L = all_channel_data.shape[1]
        Fs = 128
        data_fft = np.fft.fft(all_channel_data, axis=1)
        frequency = np.abs(data_fft / L)
        frequency = frequency[:, : L // 2 + 1]
        freq_bins = np.linspace(0, Fs / 2, L // 2 + 1)

        delta = frequency[:, (freq_bins >= 1) & (freq_bins < 4)]
        theta = frequency[:, (freq_bins >= 4) & (freq_bins < 8)]
        alpha = frequency[:, (freq_bins >= 8) & (freq_bins < 13)]
        beta = frequency[:, (freq_bins >= 13) & (freq_bins < 30)]
        gamma = frequency[:, (freq_bins >= 30) & (freq_bins < 50)]

        return delta, theta, alpha, beta, gamma

    def predict_emotion(self, feature):
        """
        Predicts arousal and valence using trained Decision Tree models.
        """
        feature = self.scaler.transform(feature)  # 스케일링 적용
        result_ar = self.tree_arousal.predict(feature)[0]
        result_va = self.tree_valence.predict(feature)[0]
        return result_ar, result_va

    def determine_emotion_class(self, feature):
        """
        Maps predicted arousal and valence to emotion classes based on Russell's Circumplex Model.
        """
        class_ar, class_va = self.predict_emotion(feature)

        if class_ar == 2.0 or class_va == 2.0:
            emotion_class = 5
        elif class_ar == 3.0 and class_va == 1.0:
            emotion_class = 1
        elif class_ar == 3.0 and class_va == 3.0:
            emotion_class = 2
        elif class_ar == 1.0 and class_va == 3.0:
            emotion_class = 3
        elif class_ar == 1.0 and class_va == 1.0:
            emotion_class = 4
        return emotion_class

    def process_all_data(self, all_channel_data):
        """
        Processes EEG data and predicts emotion class.
        """
        feature = self.get_feature(all_channel_data)
        emotion_class = self.determine_emotion_class(feature)
        print("Extracted feature vector:", feature)
        return emotion_class

    def main_process(self):
        """
        Loads EEG data, processes it, and predicts emotion.
        """
        fname = "../../DEAP/s01.mat"
        mat_data = sio.loadmat(fname)

        eeg_data = mat_data['data']
        labels = mat_data['labels']

        trial = 36
        eeg_realtime = eeg_data[trial, :, :]
        print("Labels for trial:", labels[trial, :])

        eeg_raw = eeg_realtime[:32, :]

        emotion_class = self.process_all_data(eeg_raw)
        print("Class of emotion =", emotion_class)

        emotion_dict = {
            1: "fear - nervous - stress - tense - upset",
            2: "happy - alert - excited - elated",
            3: "relax - calm - serene - contented",
            4: "sad - depressed - lethargic - fatigue",
            5: "Neutral"
        }
        print(f"Emotion: {emotion_dict.get(emotion_class, 'Unknown')}")

        return emotion_class


if __name__ == "__main__":
    rte = PredictEmotion()
    rte.main_process()