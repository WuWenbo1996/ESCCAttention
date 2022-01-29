import validate
import torch
import argparse

# Import models
# import models.resnet
import models.se_resnet
# import models.cbam_resnet

import dataloaders

DATASET = 'ESC'
DATANAME = 'ESC50'
MODELNAME = 'resnet50'
SOURCE_DATA = '/home/gaoya/data/' + DATANAME

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default=DATASET)
parser.add_argument("--dataset_name", type=str, default=DATANAME)
parser.add_argument("--data_dir", type=str, default=SOURCE_DATA+"/store/")
parser.add_argument("--model", type=str, default=MODELNAME)
parser.add_argument("--checkpoint_dir", type=str, default="se/"+MODELNAME)
args = parser.parse_args()

if __name__ == "__main__":
    last_acc = 0
    best_acc = 0

    # ----------  Configuration  -------------
    NUM_FOLD = {'USC': 10, 'ESC': 5}
    BATCH_SIZE = 1
    NUM_WORKERS = 0

    for i in range(1, NUM_FOLD[args.dataset]+1):
        val_loader = dataloaders.datasetnormal.fetch_dataloader("{}validation128mel{}.pkl".format(args.data_dir, i),
                                                                args.dataset_name, BATCH_SIZE, NUM_WORKERS)

        # model = models.resnet.resnet50(dataset=args.dataset, pretrained=False)
        model = models.se_resnet.se_resnet50(dataset=args.dataset, pretrained=False)
        # model = models.resnet.resnet50(dataset=args.dataset, pretrained=False)

        model = model.cuda()

        # best model for this fold
        checkpoint = torch.load("checkpoint/{}/{}/model_best_{}.pth.tar".format(args.dataset_name, args.checkpoint_dir, i))
        model.load_state_dict(checkpoint["model"])
        best_acc_fold = validate.evaluate(model val_loader)
        best_acc += (best_acc_fold / NUM_FOLD[args.dataset])

        # last model for this fold
        checkpoint = torch.load("checkpoint/{}/{}/last{}.pth.tar".format(args.dataset_name, args.checkpoint_dir, i))
        model.load_state_dict(checkpoint["model"])
        last_acc_fold = validate.evaluate(model, val_loader)
        last_acc += (last_acc_fold / NUM_FOLD[args.dataset])

        print("Fold {} Best Acc:{} Last Acc:{}".format(i, best_acc_fold, last_acc_fold))

    print("Dataset:{} Model:{} Best Acc:{} Last Acc:{}".format(args.dataset_name, args.model, best_acc, last_acc))

