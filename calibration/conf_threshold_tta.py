import torch
import torchvision.transforms.functional as TF
import sys
import random
import numpy as np
from scipy.special import softmax
from torch import optim
from torch.nn import functional as F
from torch.distributions import Categorical
import argparse
import os
import glob
import math
import timm

sys.path.insert(1, '../')
import datetime
from lib.dataset import load_predefined_test

seed = 1
batch_size=32

output_dir='/kaggle/working'
valid_datadir='/kaggle/tmp/eyepacs/eyepacs-multiclass/test'

parser = argparse.ArgumentParser(
                    description="Evaluate confidence calibration of neural network for ")
parser.add_argument("-d", "--dir",
                    help="directory location containing pretrained models")
parser.add_argument("-o", "--optimizers", default="",
                    help="optimizer")
parser.add_argument("-n", "--nmodel", default=10,
                    help="number of models")
parser.add_argument("-max", "--model_max_idx", default=100,
                    help="number of models")
parser.add_argument("-ds", "--dataset",
                    help="dataset")
parser.add_argument("-valdir", "--val_dir_param", default=valid_datadir,
                    help="directory containing validation data")
parser.add_argument("-r", "--rotate", default=0,
                    help="rotate data")
parser.add_argument("-out", "--output_dir", default=output_dir,
                    help="directory to store output plots")
parser.add_argument("-mint", "--min_threshold", default=0.,
                    help="minimum threshold to be applied")
parser.add_argument("-maxt", "--max_threshold", default=1.,
                    help="maximum threshold to be applied")
parser.add_argument("-ens", "--ensemble", default=False, action='store_true',
                    help="use multiple checkpoints for non-SGLD optimizers i.e ensemble")
parser.add_argument("-tta", "--tta", default=False, action='store_true',
                    help="use test-time augmentation. default is False")

args = parser.parse_args()
dir_path = str(args.dir)
dataset = str(args.dataset)
valid_datadir = str(args.val_dir_param)
optimizers = str(args.optimizers)
nmodel = int(args.nmodel)
rotate = int(args.rotate)
nmodel_max = int(args.model_max_idx)
min_threshold = float(args.min_threshold)
max_threshold = float(args.max_threshold)
ensemble = bool(args.ensemble)
tta = bool(args.tta)

optimizers = optimizers.split(",")

torch.cuda.set_device(0)
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# prepare output dir
output_dir = f'{output_dir}/conf_threshold'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

thresholds = [.0, .1,.2, .3, .4, .5, .6, .7, .8, .9, 1.]
thresholds = [x for x in thresholds if x >= min_threshold and x <= max_threshold]
accuracies = {} # key=optimizer, value=array of accuracies in threshold order
aucs = {} # key=optimizer, value=array of auc_mu in threshold order
samples = {} # key=optimizer, value=number of samples in threshold order
entropies = {} # key=optimizer, value=list of entropy

session_id_prefix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
stats_path = f'{output_dir}/stats_{session_id_prefix}.pt'
f = open(f'{output_dir}/stats_{session_id_prefix}.txt', 'w')
print(f'dataset: {dataset}', file=f)
print(f'tta: {tta}', file=f)
print(f'path: {dir_path}', file=f)
print(f'rotated: {rotate}', file=f)
print(f'statistics: {stats_path}', file=f)

for optimizer in optimizers:
    model = timm.create_model('inception_v4', pretrained=False, num_classes=3)
    model = model.cuda()

    # load data based on param `dataset`
    val_dataset = load_predefined_test(valid_datadir, batch_size=batch_size, aug=tta)

    models = []
    path_idxs = [i for i in range(nmodel_max, nmodel_max-nmodel, -1)]

    # only use last checkpoint for non SGLD:
    if optimizer in ["SGD","RMSProp","Adam"] and not ensemble:
        path_idxs = [nmodel_max]
        nmodel = 1

    path_idxs = sorted(path_idxs)

    pred_class_list = [] # list of class prediction
    pred_probs = [] # list of confidence value
    data_labels = [] # list of correct class
    loss_list = [] # list of per batch NLL loss
    entropies[optimizer] = [] # list of prediction entropy value
    correct = 0
    total = 0

    with torch.no_grad():
      for path_idx in path_idxs:
        # prepare pretrained model:
        path_glob = dir_path + f"/{optimizer}/*_*_{path_idx}.pt"
        # print("pretrained path glob: ", path_glob)
        path = glob.glob(path_glob)[0]
        print("pretrained path: ", path)
        chk = torch.load(path)
        model.load_state_dict(chk['model_state_dict'])
        model.eval()

        logits = model(data)
        prob_vecs = F.softmax(logits,dim=1) # (32, 3); (batch_size, num_class)
        mean_pred_soft += prob_vecs/float(nmodel) # (32, 3)
        mean_log_soft += F.log_softmax(logits,dim=1)/float(nmodel)

        for data, target in val_dataset:
          data = data.cuda()
          target = target.cuda()

          if rotate > 0:
              # use torchvision transform instead, much simpler: 
              # https://pytorch.org/vision/stable/transforms.html#functional-transforms
              data = TF.rotate(data, rotate)

          mean_pred_soft = torch.zeros(len(data), 3).cuda()
          mean_log_soft = torch.zeros(len(data), 3).cuda()
          
          entropies[optimizer].append(Categorical(probs = mean_pred_soft).entropy())
          pred_probs.append(mean_pred_soft.cpu())
          data_labels.append(target.cpu())
          loss = F.nll_loss(mean_log_soft, target)
          loss_list.append(loss.cpu().numpy())
            

          #total
          pred = mean_pred_soft.data.max(1)[1] 
          pred_class_list.append(pred.cpu())
          total += target.size(0)
          correct += (pred == target).sum().item()

        pred_probs_soft = torch.cat(pred_probs)
        target = torch.cat(data_labels)

    print(f"Calculate calibration for network trained using {optimizer} {nmodel} models")
    val_acc = 100 * correct / total
    print(f"Accuracy of the network on the test images rotated {rotate} degree: {val_acc:.2f}")
    print(total)

    ################
    #metrics on confidence thresholds

    accuracies[optimizer] = []
    samples[optimizer] = []
    aucs[optimizer] = []

    for thres in thresholds:
        correct = 0
        total = 0
        thres_pred_probs = []
        thres_labels = []
        for (pred_prob, pred_class, label) in zip(pred_probs,pred_class_list,data_labels):
            class_pred_prob = pred_prob.data.max(1)[0]
            # find tensor indices where class predictive confidence is above threshold
            # https://stackoverflow.com/a/57570139/2998271
            mask = class_pred_prob >= thres
            indices = torch.nonzero(mask)
            
            # update counter using valid indices only:
            total += label[indices].size(0)
            correct += (pred_class[indices] == label[indices]).sum().item()

            thres_pred_probs.append(pred_prob[indices])
            thres_labels.append(label[indices])


        val_acc = 0
        if total > 0:
            val_acc = 100 * correct / total

        # save accuracy and number of samples:
        accuracies[optimizer].append(val_acc)
        samples[optimizer].append(total)

torch.save({
            'thresholds': thresholds,
            'accuracies': accuracies,
            'entropies': entropies,
            'samples': samples
        }, stats_path)

f.close()