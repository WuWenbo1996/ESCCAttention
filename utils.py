import shutil
import os
import torch
import torch.nn as nn


class RunningLoss:
    def __init__(self):
        self.total = 0
        self.steps = 0

    def update(self, score):
        self.total += score
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


class RunningAcc:
    def __init__(self):
        self.total = 0
        self.num = 0

    def update(self, score, sample_num):
        self.total += score
        self.num += sample_num

    def __call__(self):
        return self.total / float(self.num)


def save_checkpoint(state, is_best, split, checkpoint):
    filename = os.path.join(checkpoint, 'last{}.pth.tar'.format(split))
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist")
        os.makedirs(checkpoint)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(checkpoint, "model_best_{}.pth.tar".format(split)))


def load_checkpoint(checkpoint, model, optimizer=None, parallel=False):
    if not os.path.exists(checkpoint):
        raise ("File Not Found Error {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    if parallel:
        model.module.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint["model"])

    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint


def initialize_weights(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Linear') != -1:
        nn.init.ones_(m.weight.data)
