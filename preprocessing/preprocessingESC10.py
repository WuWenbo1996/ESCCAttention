# log-spectrogram
# Load -> STFT -> abs -> power -> log
import os
import librosa
import librosa.display
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch

# Hyper Params
DATA_DIR = "E:/dataset/"
ESC10_AUDIO_PATH = os.path.join(DATA_DIR, 'esc/audio/')
ESC10_METADATA_PATH = os.path.join(DATA_DIR, 'esc/meta/esc50.csv')
STORE_PATH = os.path.join(DATA_DIR, 'melspectrogram/')
PKL_PATH = os.path.join(STORE_PATH, 'esc10_df.pkl')

SAMPLE_RATE = 44100
AUDIO_LENGTH = 5

"""
windows size: 1024 ms
hop size: 512 ms
mel filter banks: 128
"""

# audio clip: 50% overlap
WINDOW_LENGTH = 1024  # 23.2 ms
HOP_LENGTH = 512  # 11.6 ms
N_MEL = 128  # 128 mel bins


def compute_melspectrogram(audio):
    # Zero-padding or truncate for ESC-10, fixed to 5s
    if len(audio) > SAMPLE_RATE * AUDIO_LENGTH:
        audio = audio[:SAMPLE_RATE * AUDIO_LENGTH]

    if len(audio) < SAMPLE_RATE * AUDIO_LENGTH:
        audio = np.append(audio, [0.0] * (SAMPLE_RATE * AUDIO_LENGTH - len(audio)))

    """Visualize waveform"""
    # import matplotlib.pyplot as plt
    # plt.figure()
    # librosa.display.waveplot(audio, SAMPLE_RATE)
    # plt.title('Beat wavform')
    # plt.show()


    """compute a mel-scaled spectrogram"""
    # n_fft是窗口大小，hop_length是相邻窗口之间的距离，此处相邻窗之间有50%的overlap，n_mels是mel bands的数量
    melspectrogram = librosa.feature.melspectrogram(y=audio,
                                                    sr=SAMPLE_RATE,
                                                    n_fft=WINDOW_LENGTH,
                                                    hop_length=HOP_LENGTH,
                                                    n_mels=N_MEL)

    """convert a power spectrogram to decibel units (log-mel spectrogram)"""
    # 音频信号的时频表示特征
    log_melspectrogram = librosa.power_to_db(melspectrogram)
    log_melspectrogram_delta = librosa.feature.delta(log_melspectrogram, order=1)
    log_melspectrogram_delta_delta = librosa.feature.delta(log_melspectrogram, order=2)

    log_melspectrogram = (log_melspectrogram - np.mean(log_melspectrogram)) / np.max(np.abs(log_melspectrogram))
    log_melspectrogram_delta = (log_melspectrogram_delta - np.mean(log_melspectrogram_delta)) / np.max(np.abs(log_melspectrogram_delta))
    log_melspectrogram_delta_delta = (log_melspectrogram_delta_delta - np.mean(log_melspectrogram_delta_delta)) / np.max(np.abs(log_melspectrogram_delta_delta))

    log_melspectrogram_comb = np.stack([log_melspectrogram, log_melspectrogram_delta, log_melspectrogram_delta_delta])

    """Visualize spec"""
    import matplotlib.pyplot as plt
    plt.subplot(4, 1, 1)
    librosa.display.specshow(log_melspectrogram)
    plt.subplot(4, 1, 2)
    librosa.display.specshow(log_melspectrogram_delta)
    plt.title(r'MFCC-$\Delta$')
    plt.colorbar()
    plt.subplot(4, 1, 3)
    librosa.display.specshow(log_melspectrogram_delta_delta, x_axis='time')
    plt.title(r'MFCC-$\Delta^2$')
    plt.colorbar()
    plt.tight_layout()
    plt.subplot(4, 1, 4)
    plt.imshow(log_melspectrogram_comb.transpose(1,2,0))
    plt.show()

    # Save Samples(Origin Sound, mel spectrogram of origin sound, label)
    print(log_melspectrogram_comb)
    return log_melspectrogram_comb

if __name__ == "__main__":
    esc10_metadata_df = pd.read_csv(ESC10_METADATA_PATH,
                                    usecols=["filename", "fold", "target", "esc10"],
                                    dtype={"fold": "uint8", "target": "uint8"})

    esc10_metadata_df = esc10_metadata_df[esc10_metadata_df["esc10"]]

    target = list(set(list(esc10_metadata_df['target'])))

    new_target = {target[i]:i for i in range(10)}

    features = []

    if not os.path.exists(STORE_PATH):
        print("Mel Spectrogram Store Directory does not exist")
        os.makedirs(STORE_PATH)

    # iterate through all dataset examples and compute log-mel spectrograms
    for index, row in tqdm(esc10_metadata_df.iterrows(), total=len(esc10_metadata_df)):
        file_path = f'{ESC10_AUDIO_PATH}/{row["filename"]}'
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)

        log_melspectrogram_comb = compute_melspectrogram(audio)
        label = new_target[row["target"]]
        fold = row["fold"]

        features.append([log_melspectrogram_comb, label, fold])

    # convert into a Pandas DataFrame
    esc10_df = pd.DataFrame(features, columns=["mel_spec", "label", "fold"])

    esc10_df.to_pickle(PKL_PATH)
