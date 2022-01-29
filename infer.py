import validate
import torch
import argparse

# Import models
import models.resnet

import dataloaders

import json
import threading

import socket

HOST = '' # Symbolic name meaning all available interfaces
PORT = 9494 # Arbitrary non-privileged port
BUFSIZE = 1024

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='ESC10')
parser.add_argument("--dataset_name", type=str, default='ESC')
parser.add_argument("--model", type=str, default='resnet50')
parser.add_argument("--checkpoint", type=str, default='checkpoint/ESC10/base/model_best_1.pth.tar')
args = parser.parse_args()

class Thread_recv_data(threading.Thread):
    def __init__(self, model, host, port):
        super().__init__()
        self.host = host
        self.port = port
        self.model = model
        self.server = None

        # data
        self.recv_msg, self.client_addr = None, None

    def create_socket(self):
        try:
            self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        except socket.error as msg:
            print('Failed to create socket. Error message: ' , msg)
            sys.exit()

        print('socket created')

    def bind_socket(self):
        try:
            self.server.bind((HOST, PORT))
        except socket.error as msg:
            print('Bind failed. Error msg:' , msg)
            sys.exit()

        print('socket bind complete')

    def listen_socket(self):
        try:
            self.server.listen(128)
        except socket.error as msg:
            print('Listen failed. Error msg:' , msg)
            sys.exit()

    def compute_melspectrogram(self):
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

    def run(self):

        self.create_socket()
        self.bind_socket()
        self.listen_socket()

        # Function for handling connections. This will be used to create threads
        while True:
            client_socket, self.client_addr = self.server.accept()

            while True:
                self.recv_msg = client_socket.recv(BUFSIZE)
                if self.recv_msg:
                    print(self.recv_msg.decode())
                else:
                    break
            client_socket.close()


if __name__ == "__main__":
    model = models.resnet.resnet50(dataset=args.dataset, pretrained=False)
    model = model.cuda()
    # best model for this fold
    checkpoint = torch.load(args.checkpoint)

    # model.load_state_dict(checkpoint["model"])

    thread_data = Thread_recv_data(model, HOST, PORT)
    thread_data.start()
