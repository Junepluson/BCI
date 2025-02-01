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
import scipy.spatial as ss
import scipy.stats as sst
import _pickle as cPickle
from pathlib import Path
import itertools
import random
import cv2
import scipy.io as sio


sampling_rate = 128
number_of_channel = 32  # considering only head electrodes
eeg_in_second = 63  # length of each trial
number_of_eeg = sampling_rate * eeg_in_second  # total inputs from a single channel

channel_names = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz',
                 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2']


class predictEmotion(object):
    """
    Receives EEG data preprocessing and predict emotion.
    """

    # path is set to training data directory
    def __init__(self):
        """
        Initializes training data and their classes.
        """
        self.train_arousal = self.get_csv("train_std.csv")
        self.train_valence = self.get_csv("train_std.csv")
        self.class_arousal = self.get_csv("class_arousal.csv")
        self.class_valence = self.get_csv("class_valence.csv")

    def get_csv(self, path):
        """
        Get data from csv and convert them to numpy python.
        Input: Path csv file.
        Output: Numpy array from csv data.
        """
        # Get csv data to list
        file_csv = open(path)
        data_csv = csv.reader(file_csv)
        # convert list to array with a specific dtype
        data_training = np.array(list(data_csv), dtype=np.float64)
        return data_training

    def do_fft(self, all_channel_data):
        """
        Do fft in each channel for all channels.
        Input: Channel data with dimension N x M. N denotes number of channel and M denotes number of EEG data from each channel.
        Output: FFT result with dimension N x M. N denotes number of channel and M denotes number of FFT data from each channel.
        """
        data_fft = map(lambda x: np.fft.fft(x), all_channel_data)

        return data_fft

    def get_frequency(self, all_channel_data):
        """
        Get frequency from computed fft for all channels.
        Input: Channel data with dimension N x M. N denotes number of channel and M denotes number of EEG data from each channel.
        Output: Frequency band from each channel: Delta, Theta, Alpha, Beta, and Gamma.
        """
        # Length data channel
        L = len(all_channel_data[0])

        # Sampling frequency
        Fs = 128

        # Get fft data
        data_fft = self.do_fft(all_channel_data)

        # Compute frequency
        frequency = map(lambda x: abs(x // L), data_fft)
        frequency = map(lambda x: x[: L // 2 + 1] * 2, frequency)

        # creating 5 instances of frequency iterator
        f1, f2, f3, f4, f5 = itertools.tee(frequency, 5)

        # List frequency
        delta = np.array(list(map(lambda x: x[L * 1 // Fs - 1: L * 4 // Fs], f1)))
        theta = np.array(list(map(lambda x: x[L * 4 // Fs - 1: L * 8 // Fs], f2)))
        alpha = np.array(list(map(lambda x: x[L * 5 // Fs - 1: L * 13 // Fs], f3)))
        beta = np.array(list(map(lambda x: x[L * 13 // Fs - 1: L * 30 // Fs], f4)))
        gamma = np.array(list(map(lambda x: x[L * 30 // Fs - 1: L * 50 // Fs], f5)))

        return delta, theta, alpha, beta, gamma

    # def get_feature(self, all_channel_data):
    #     """
    #     Get feature from each frequency.
    #     Input: Channel data with dimension N x M. N denotes number of channel and M denotes number of EEG data from each channel.
    #     Output: Feature (standard deviasion and mean) from all frequency bands and channels with dimesion 1 x M (number of feature).
    #     """
    #
    #     (delta, theta, alpha, beta, gamma) = self.get_frequency(all_channel_data)
    #     delta_std = np.std(delta, axis=1)
    #     theta_std = np.std(theta, axis=1)
    #     alpha_std = np.std(alpha, axis=1)
    #     beta_std = np.std(beta, axis=1)
    #     gamma_std = np.std(gamma, axis=1)
    #     feature = np.array([delta_std, theta_std, alpha_std, beta_std, gamma_std])
    #     feature = feature.T
    #     feature = feature.ravel()
    #
    #     return feature
    def get_feature(self, all_channel_data): #수정
        """
        Get feature from each frequency.
        Input: Channel data with dimension N x M. N denotes number of channel and M denotes number of EEG data from each channel.
        Output: Feature (standard deviation and mean) from all frequency bands and channels with dimension 1 x M (number of features).
        """
        # Extract frequency bands
        delta, theta, alpha, beta, gamma = self.get_frequency(all_channel_data)

        # Compute standard deviation for each frequency band
        delta_std = np.std(delta, axis=1)
        theta_std = np.std(theta, axis=1)
        alpha_std = np.std(alpha, axis=1)
        beta_std = np.std(beta, axis=1)
        gamma_std = np.std(gamma, axis=1)

        # Combine all features into a single array
        feature = np.concatenate([delta_std, theta_std, alpha_std, beta_std, gamma_std])
        return feature

    def get_frequency(self, all_channel_data):
        """
        Get frequency bands (Delta, Theta, Alpha, Beta, Gamma) from EEG data.
        Input: Channel data (N x M), where N is the number of channels and M is the number of samples.
        Output: Frequency bands for each channel: Delta, Theta, Alpha, Beta, Gamma.
        """
        # Number of samples (length of data in time)
        L = all_channel_data.shape[1]

        # Sampling frequency
        Fs = 128

        # Get FFT data
        data_fft = np.fft.fft(all_channel_data, axis=1)  # FFT along the sample axis

        # Compute frequency magnitudes
        frequency = np.abs(data_fft / L)  # Normalize by the length of the data
        frequency = frequency[:, : L // 2 + 1]  # Keep only positive frequencies

        # Frequency bins
        freq_bins = np.linspace(0, Fs / 2, L // 2 + 1)  # Frequency values for bins

        # Extract frequency bands
        delta = frequency[:, (freq_bins >= 1) & (freq_bins < 4)]
        theta = frequency[:, (freq_bins >= 4) & (freq_bins < 8)]
        alpha = frequency[:, (freq_bins >= 8) & (freq_bins < 13)]
        beta = frequency[:, (freq_bins >= 13) & (freq_bins < 30)]
        gamma = frequency[:, (freq_bins >= 30) & (freq_bins < 50)]

        return delta, theta, alpha, beta, gamma
    def predict_emotion(self, feature):
        """
        Get arousal and valence class from feature.
        Input: Feature (standard deviasion and mean) from all frequency bands and channels with dimesion 1 x M (number of feature).
        Output: Class of emotion between 1 to 3 from each arousal and valence. 1 denotes low , 2 denotes neutral, and 3 denotes high .
        """
        # Compute canberra with arousal training data
        distance_ar = list(map(lambda x: ss.distance.canberra(x, feature), self.train_arousal))
        # Compute canberra with valence training data
        distance_va = list(map(lambda x: ss.distance.canberra(x, feature), self.train_valence))

        # Compute 3 nearest index and distance value from arousal
        idx_nearest_ar = np.array(np.argsort(distance_ar)[:3])
        val_nearest_ar = np.array(np.sort(distance_ar)[:3])

        # Compute 3 nearest index and distance value from arousal
        idx_nearest_va = np.array(np.argsort(distance_va)[:3])
        val_nearest_va = np.array(np.sort(distance_va)[:3])

        # Compute comparation from first nearest and second nearest distance. If comparation less or equal than 0.7, then take class from the first nearest distance. Else take frequently class.
        # Arousal
        comp_ar = val_nearest_ar[0] / val_nearest_ar[1]
        if comp_ar <= 0.7:
            result_ar = self.class_arousal[0, idx_nearest_ar[0]]
        else:
            result_ar = sst.mode(self.class_arousal[0, idx_nearest_ar])
            result_ar = float(result_ar[0])

        # Valence
        comp_va = val_nearest_va[0] / val_nearest_va[1]
        if comp_va <= 0.7:
            result_va = self.class_valence[0, idx_nearest_va[0]]
        else:
            result_va = sst.mode(self.class_valence[0, idx_nearest_va])
            result_va = float(result_va[0])

        return result_ar, result_va

    def determine_emotion_class(self, feature):
        """
        Get emotion class from feature.
        Input: Feature (standard deviasion) from all frequency bands and channels with dimesion 1 x M (number of feature).
        Output: Class of emotion between 1 to 5 according to Russel's Circumplex Model.
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
        Process all data from EEG data to predict emotion class.
        Input: Channel data with dimension N x M. N denotes number of channel and M denotes number of EEG data from each channel.
        Output: Class of emotion between 1 to 5 according to Russel's Circumplex Model. And send it to web ap
        """
        # Get feature from EEG data
        feature = self.get_feature(all_channel_data)

        # Predict emotion class
        emotion_class = self.determine_emotion_class(feature)
        print("Extracted feature vector:", feature)
        return emotion_class

    #
    def send_result_to_window(self, emotion_class):
        """
        Send emotion predict to web app.
        Input: Class of emotion between 1 to 5 according to Russel's Circumplex Model.
        Output: Send emotion prediction to web window.
        """
        i1 = cv2.imread('emoji/1.png')
        i2 = cv2.imread('emoji/2.png')
        i3 = cv2.imread('emoji/3.png')
        i4 = cv2.imread('emoji/4.png')
        i5 = cv2.imread('emoji/5.png')
        if emotion_class == 1:
            cv2.imshow('image', i1)
        elif emotion_class == 2:
            cv2.imshow('image', i2)
        elif emotion_class == 3:
            cv2.imshow('image', i3)
        elif emotion_class == 4:
            cv2.imshow('image', i4)
        else:
            cv2.imshow('image', i5)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def main_process(self):
        """
        Input: Get EEG data from DEAP, process all data (FFT, feature extraction, and classification), and predict the emotion.
        Output: Class of emotion between 1 to 5 according to Russel's Circumplex Model.
        """
        # Load MATLAB .mat file
        fname = "../../DEAP/s03.mat"
        mat_data = sio.loadmat(fname)

        # Extract EEG data and labels
        eeg_data = mat_data['data']  # Update this key based on your .mat file structure
        labels = mat_data['labels']  # Update this key based on your .mat file structure

        # Check the structure of the loaded data
        print("Loaded EEG data shape:", eeg_data.shape)
        print("Loaded labels shape:", labels.shape)

        # Example: Use the 10th trial
        trial = 3 # Adjust trial index as needed
        eeg_realtime = eeg_data[trial, :, :]  # Assuming data is in [trials, channels, time] format
        print("Labels for trial:", labels[trial, :])  # Print labels for the selected trial

        # Use only the first 32 channels
        eeg_raw = eeg_realtime[:32, :]
        print("Raw EEG data shape (first 32 channels):", eeg_raw.shape)

        # Process and predict emotion
        emotion_class = self.process_all_data(eeg_raw)
        print("Class of emotion =", emotion_class)
        print(mat_data.keys())
        print(mat_data['data'].shape)
        print(mat_data['labels'].shape)
        # Map emotion class to description
        if emotion_class == 1:
            print("Emotion: fear - nervous - stress - tense - upset")
        elif emotion_class == 2:
            print("Emotion: happy - alert - excited - elated")
        elif emotion_class == 3:
            print("Emotion: relax - calm - serene - contented")
        elif emotion_class == 4:
            print("Emotion: sad - depressed - lethargic - fatigue")
        else:
            print("Emotion: Neutral")

        # Display result in a window
        self.send_result_to_window(emotion_class)


#

if __name__ == "__main__":
    rte = predictEmotion()
    rte.main_process()
