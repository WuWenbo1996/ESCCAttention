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
IMG_PATH = os.path.join(STORE_PATH, 'esc10_img/')

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


def compute_melspectrogram(audio, file_name):
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
    melspectrogram = librosa.feature.melspectrogram(y=audio,
                                                    sr=SAMPLE_RATE,
                                                    hop_length=HOP_LENGTH,
                                                    win_length=WINDOW_LENGTH,
                                                    n_mels=N_MEL)

    """convert a power spectrogram to decibel units (log-mel spectrogram)"""
    melspectrogram = librosa.power_to_db(melspectrogram)

    import matplotlib.pyplot as plt
    plt.figure()
    fig, ax = plt.subplots()
    plt.imshow(melspectrogram)
    file_name = IMG_PATH + file_path.split('/')[-1] + ".jpg"
    plt.axis("off")
    plt.xticks([])
    plt.yticks([])
    fig.set_size_inches(melspectrogram.shape[1] / 100.0, melspectrogram.shape[0] / 100.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
    plt.margins(0,0)
    plt.savefig(file_name)
    plt.close()

    # if file_name == "E:/dataset/melspectrogram/esc10_img/5-212181-A-38.wav.jpg":
    #     print(np.sum(melspectrogram,0))
    #     print(np.sum(melspectrogram,1))
    #     print(melspectrogram.shape)
    # Normalization
    # melspectrogram = (melspectrogram - np.mean(melspectrogram)) / np.std(melspectrogram)

    """Visualize spec"""
    # import matplotlib.pyplot as plt
    # plt.subplot(4, 1, 1)
    # librosa.display.specshow(melspectrogram)
    # plt.subplot(4, 1, 2)
    # librosa.display.specshow(melspectrogram_delta)
    # plt.title(r'MFCC-$\Delta$')
    # plt.colorbar()
    # plt.subplot(4, 1, 3)
    # librosa.display.specshow(melspectrogram_delta2, x_axis='time')
    # plt.title(r'MFCC-$\Delta^2$')
    # plt.colorbar()
    # plt.tight_layout()
    # plt.subplot(4, 1, 4)
    # plt.imshow(specs)
    # plt.show()
    # print(specs)

    # Save Samples(Origin Sound, mel spectrogram of origin sound, label)
    return file_name

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

    if not os.path.exists(IMG_PATH):
        print("Mel Spectrogram IMG Store Directory does not exist")
        os.makedirs(IMG_PATH)

    # iterate through all dataset examples and compute log-mel spectrograms
    for index, row in tqdm(esc10_metadata_df.iterrows(), total=len(esc10_metadata_df)):
        file_path = f'{ESC10_AUDIO_PATH}/{row["filename"]}'
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)

        img_file_name = compute_melspectrogram(audio, file_path)
        label = new_target[row["target"]]
        fold = row["fold"]

        features.append([img_file_name, label, fold])

    # convert into a Pandas DataFrame
    esc10_df = pd.DataFrame(features, columns=["file_name", "label", "fold"])

    esc10_df.to_pickle(PKL_PATH)

