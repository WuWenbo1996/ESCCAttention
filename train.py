"""
Train models
"""
import os
import numpy as np
import random
import pandas as pd

import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import torchvision.transforms as transforms

# Import Dataloaders
from torch.utils.data import Dataset, DataLoader
from dataloaders import AudioDataset, FrequencyMask, TimeMask

DATA_NAME = 'ESC10'
FOLD = {'ESC10':5, 'ESC50':5, 'USC':10}
MODEL_NAME = 'base'

# Import models
if MODEL_NAME == 'base':
    import models.resnet
elif MODEL_NAME == 'se':
    import models.se_resnet
elif MODEL_NAME =='cbam':
    import models.cbam_resnet

import utils
import validate

# Hyper Params
parser = argparse.ArgumentParser(description='PyTorch ESC(ESC-10|ESC-50)/USC Training')
parser.add_argument('--dataset_name', default=DATA_NAME, type=str, help='Name of Dataset(ESC10/ESC50/USC)')
parser.add_argument('--pkl_dir', default='E:/dataset/melspectrogram/esc10_df.pkl', type=str, help='Path where the spectrograms are stored')
parser.add_argument('--fold_num', default=FOLD[DATA_NAME], type=int, help='Number of database fold')
parser.add_argument('--pretrained', default=True, type=bool, help='true/false')
parser.add_argument('--batch_size', default=32, type=int, help='Batch Size(32)')
parser.add_argument('--num_workers', default=0, type=int, help='Number of CPU Cores for DataLoaders')
parser.add_argument('--epochs', default=70, type=int, help='Number of Epochs')
parser.add_argument('--lr', default=0.0001, type=float, help='Learning Rate')
parser.add_argument('--weight_decay', default=0.01, type=float, help='Weight Decay')
parser.add_argument('--checkpoint_dir', default='checkpoint/'+DATA_NAME+'/'+MODEL_NAME+'/', type=str, help='path to store checkpoint')
parser.add_argument('--process_dir', default='process/'+DATA_NAME+'/'+MODEL_NAME+'/', type=str, help='path to training process')
parser.add_argument('--seed', default=0, type=int, help='random seed')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# random seed settings
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(args.seed)

# Training
def train(model, train_loader, optimizer, loss_fn):
    model.train()

    # Record training loss and training accuracy
    loss_avg = utils.RunningLoss()
    acc_avg = utils.RunningAcc()

    with tqdm(total=len(train_loader)) as t:
        for batch_idx, data in enumerate(train_loader):
            inputs = data['spectrogram'].cuda()

            # import matplotlib.pyplot as plt
            # import numpy as np
            # img_inputs = np.array(inputs.cpu()[0])
            # plt.imshow(img_inputs.transpose((1,2,0)))
            # plt.show()

            target = data['label'].cuda()

            outputs = model(inputs)
            loss = loss_fn(outputs, target)
            _, predicted = torch.max(outputs.data, 1)
            acc = (predicted == target).sum().item()

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_avg.update(loss.item())
            acc_avg.update(acc, target.size(0))

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()), acc='{:05.5f}'.format(acc_avg()))
            t.update()

    # Print loss of entire loss process
    return loss_avg(), acc_avg()


if __name__ == '__main__':
    # transforms
    train_transforms = transforms.Compose({
        FrequencyMask(max_width=10, use_mean=False),
        TimeMask(max_width=10, use_mean=False),
        transforms.ToTensor()
    })

    val_transforms = transforms.Compose({
        transforms.ToTensor()
    })

    audio_df = pd.read_pickle(args.pkl_dir)

    acc_set = np.zeros(args.fold_num)

    for i in range(1, args.fold_num + 1):
        # Data
        history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

        print('==> Preparing data..')
        '''spilt the data'''
        train_df = audio_df[audio_df['fold'] != i]
        val_df = audio_df[audio_df['fold'] == i]

        train_loader = DataLoader(AudioDataset(train_df, transforms=train_transforms),
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  pin_memory=True,
                                  num_workers=args.num_workers)

        val_loader = DataLoader(AudioDataset(val_df, transforms=val_transforms),
                                batch_size=args.batch_size,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=args.num_workers)

        # Model
        print('==> Building model..')
        if MODEL_NAME == 'base':
            model = models.resnet.resnet50(args.dataset_name, pretrained=args.pretrained)
        elif MODEL_NAME == 'se':
            model = models.se_resnet.se_resnet50(args.dataset_name, pretrained=args.pretrained)
        elif MODEL_NAME == 'cbam':
            model = models.cbam_resnet.cbam_resnet50(args.dataset_name, pretrained=args.pretrained)

        model = nn.DataParallel(model).cuda()

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.1)

        # ---------------- Start training ----------------
        best_acc = 0.0

        for epoch in range(1, args.epochs+1):
            # Print average loss of train process in this epoch
            avg_loss, avg_acc = train(model, train_loader, optimizer, loss_fn)

            # Accuracy of the model to validation data
            val_acc, val_loss = validate.evaluate(model, val_loader, loss_fn)
            print("Dataset {} Fold {} Epoch {}/{} Loss:{:05.3f} Acc:{:05.5f} "
                  "Valid Loss:{:05.3f} Valid Acc:{:05.5f} Best Acc:{:05.5f}".format(args.dataset_name,
                                                                                    i, epoch, args.epochs,
                                                                                    avg_loss, avg_acc, val_loss,
                                                                                    val_acc, best_acc))
            history['loss'].append(avg_loss)
            history['accuracy'].append(avg_acc)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)

            # Record best accuracy
            is_best = (val_acc > best_acc)
            if is_best:
                best_acc = val_acc
                acc_set[i - 1] = best_acc

            scheduler.step()

            # Explanation: split is idx of now training fold
            # Save last epoch and best epoch of training process
            utils.save_checkpoint({"epoch": epoch + 1,
                                   "model": model.state_dict(),
                                   "optimizer": optimizer.state_dict()}, is_best, i,
                                   "{}".format(args.checkpoint_dir))

        """Show accuracy and loss graphs for train and test sets."""
        print("Accuracy of each fold ", acc_set)
        import matplotlib.pyplot as plt

        plt.figure(figsize=(15, 5))
        plt.subplot(121)
        plt.plot(history['accuracy'])
        plt.plot(history['val_accuracy'])
        plt.grid(linestyle='--')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'], loc='upper left')

        plt.subplot(122)
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.grid(linestyle='--')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'], loc='upper left')

        if not os.path.exists(args.process_dir):
            print("Processing Directory does not exist")
            os.makedirs(args.process_dir)

        result_path = args.process_dir + str(i) + ".png"
        plt.savefig(result_path)

        """Save accuracy and loss data for train and test sets."""
        import csv
        csv_path = args.process_dir + str(i) + ".csv"
        with open(csv_path, 'w') as f:
            csv_writer = csv.writer(f)
            data_header = ["accuracy", "val_accuracy", "loss", "val_loss"]
            csv_writer.writerow(data_header)
            data = [history['accuracy'], history['val_accuracy'], history['loss'], history['val_loss']]
            data = list(zip(*data))
            csv_writer.writerows(data)

            res_header = ["best_acc"]
            csv_writer.writerow(res_header)
            res = [best_acc]
            csv_writer.writerow(res)

    print("Accuracy of each fold ", acc_set)
    print("Mean accuracy of the dataset: ", np.mean(acc_set))
