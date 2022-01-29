import os
import librosa
import librosa.display
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch

# Hyper Params
DATA_DIR = "E:/dataset/"
USC_AUDIO_PATH = os.path.join(DATA_DIR, 'UrbanSound8K/audio/')
USC_METADATA_PATH = os.path.join(DATA_DIR, 'UrbanSound8K/metadata/UrbanSound8K.csv')
STORE_PATH = os.path.join(DATA_DIR, 'melspectrogram/')
PKL_PATH = os.path.join(STORE_PATH, 'usc_df.pkl')
IMG_PATH = os.path.join(STORE_PATH, 'usc_img/')

SAMPLE_RATE = 44100
AUDIO_LENGTH = 4

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
    # Zero-padding or truncate for USC, fixed to 4s
    if len(audio) > SAMPLE_RATE * AUDIO_LENGTH:
        audio = audio[:SAMPLE_RATE * AUDIO_LENGTH]

    if len(audio) < SAMPLE_RATE * AUDIO_LENGTH:
        audio = np.append(audio, [0.0] * (SAMPLE_RATE * AUDIO_LENGTH - len(audio)))

    """Visualize waveform"""
    # import matplotlib.pyplot as plt
    # plt.figure()
    # librosa.display.waveplot(clip, SAMPLE_RATE)
    # plt.title('Beat wavform')
    # plt.show()

    # compute a mel-scaled spectrogram
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

    """Visualize spec"""
    # import matplotlib.pyplot as plt
    # plt.subplot(3, 1, 1)
    # librosa.display.specshow(melspectrogram)
    # plt.subplot(3, 1, 2)
    # librosa.display.specshow(melspectrogram_delta)
    # plt.title(r'MFCC-$\Delta$')
    # plt.colorbar()
    # plt.subplot(3, 1, 3)
    # librosa.display.specshow(melspectrogram_delta2, x_axis='time')
    # plt.title(r'MFCC-$\Delta^2$')
    # plt.colorbar()
    # plt.tight_layout()
    # plt.show()


    """Visualize SpecAug"""
    # from specAugment import spec_augment_pytorch
    # new_spectrogram = spec_augment_pytorch.spec_augment(mel_spectrogram=torch.tensor(melspectrogram_db)).squeeze(0).numpy()
    # import matplotlib.pyplot as plt
    # plt.imshow(new_spectrogram)
    # plt.show()

    # Save Samples(Origin Sound, mel spectrogram of origin sound, label)
    return file_name

if __name__ == "__main__":
    usc_metadata_df = pd.read_csv(USC_METADATA_PATH,
                                  usecols=["slice_file_name", "fold", "classID"],
                                  dtype={"fold": "uint8", "classID": "uint8"})


    features = []

    if not os.path.exists(STORE_PATH):
        print("Mel Spectrogram Store Directory does not exist")
        os.makedirs(STORE_PATH)

    if not os.path.exists(IMG_PATH):
        print("Mel Spectrogram IMG Store Directory does not exist")
        os.makedirs(IMG_PATH)

    # iterate through all dataset examples and compute log-mel spectrograms
    for index, row in tqdm(usc_metadata_df.iterrows(), total=len(usc_metadata_df)):
        file_path = f'{USC_AUDIO_PATH}/fold{row["fold"]}/{row["slice_file_name"]}'
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)

        img_file_name = compute_melspectrogram(audio, file_path)
        label = row["classID"]
        fold = row["fold"]

        features.append([img_file_name, label, fold])

    # convert into a Pandas DataFrame
    usc_df = pd.DataFrame(features, columns=["file_name", "label", "fold"])

    usc_df.to_pickle(PKL_PATH)
