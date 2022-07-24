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

import metrics
import visualization
import auc_mu

sys.path.insert(1, '../')
import timm
from lib.dataset import load_predefined_test
import datetime
from samsung_utils import Logger, Arguments
import time
from scipy.special import logsumexp

seed = 1
batch_size=32

valid_datadir='/kaggle/tmp/eyepacs/eyepacs-multiclass/test'
output_dir='/kaggle/working'

parser = argparse.ArgumentParser(
                    description="Evaluate calibrated log-likelihood of neural network for ")
parser.add_argument("-d", "--dir",
                    help="directory location containing pretrained models")
parser.add_argument("-o", "--optimizers", default="",
                    help="optimizer")
parser.add_argument("-n", "--nmodel", default=10,
                    help="number of models")
parser.add_argument("-max", "--model_max_idx", default=100,
                    help="number of models")
parser.add_argument("-ds", "--dataset", default='eyepacs',
                    help="dataset")
parser.add_argument("-r", "--rotate", default=0,
                    help="rotate data")
parser.add_argument("-valdir", "--val_dir_param", default=valid_datadir,
                    help="directory containing validation data")
parser.add_argument("-out", "--output_dir", default=output_dir,
                    help="directory to store output plots")
parser.add_argument("-ens", "--ensemble", default=False, action='store_true',
                    help="use multiple checkpoints for non-SGLD optimizers i.e ensemble")

args = parser.parse_args()
dir_path = str(args.dir)
dataset = str(args.dataset)
optimizers = str(args.optimizers)
nmodel = int(args.nmodel)
rotate = int(args.rotate)
nmodel_max = int(args.model_max_idx)
valid_datadir = str(args.val_dir_param)
ensemble = bool(args.ensemble)

optimizers = optimizers.split(",")

torch.cuda.set_device(0)
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# prepare output dir
plots_dir = f'{output_dir}/plots'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

session_id_prefix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
f = open(f'{plots_dir}/evaluation_{session_id_prefix}.txt', 'w+')
print(f'dataset: {dataset}', file=f)
print(f'path: {dir_path}', file=f)
print(f'rotated: {rotate}', file=f)

print(f"Dataset/Model\tOptimiser\tCLL", file=f)

def get_targets(loader):
    targets = []
    for _, target in loader:
        targets += [target]
    targets = np.concatenate(targets)

    return targets

def one_sample_pred(loader, model, **kwargs):
    preds = []
    model.eval()

    for input, target in loader:
        input = input.cuda()
        with torch.no_grad():
            output = model(input, **kwargs)
        log_probs = torch.nn.functional.log_softmax(output, dim=1)
        preds.append(log_probs.cpu().data.numpy())

    return np.vstack(preds)

for optimizer in optimizers:
    model = timm.create_model('inception_v4', pretrained=False, num_classes=3)
    model = model.cuda()

    # load data based on param `dataset`
    val_dataset = load_predefined_test(valid_datadir, batch_size=batch_size)
    targets = get_targets(val_dataset)

    models = []
    path_idxs = [i for i in range(nmodel_max, nmodel_max-nmodel, -1)]

    # only use last checkpoint for non SGLD:
    if optimizer in ["SGD","RMSProp","Adam"] and not ensemble:
        path_idxs = [nmodel_max]
        nmodel = 1

    path_idxs = sorted(path_idxs)
    log_probs = []
    logger = Logger(base=f'{output_dir}/logs')
    s_args = Arguments(dataset=args.dataset, model='inception_v4', method=optimizer)
    for path_idx in path_idxs:
        start = time.time()
        path_glob = dir_path + f"/{optimizer}/*_*_{path_idx}.pt"
        # print("pretrained path glob: ", path_glob)
        path = glob.glob(path_glob)[0]
        print("pretrained path: ", path)
        chk = torch.load(path)
        model.load_state_dict(chk['model_state_dict'])

        model.eval()

        ones_log_prob = one_sample_pred(val_dataset, model)
        log_probs.append(ones_log_prob)
        logger.add_metrics_ts(path_idx, log_probs, targets, s_args, time_=start)
        logger.save(s_args)

    os.makedirs(f'{output_dir}/.megacache', exist_ok=True)
    logits_pth = f'{output_dir}/logits_%s-%s-%s-%s-%s'
    logits_pth = logits_pth % (s_args.dataset, s_args.model, s_args.method, path_idx+1, 1)
    log_prob = logsumexp(np.dstack(log_probs), axis=2) - np.log(path_idx+1)

    print('Save final logprobs to %s' % logits_pth, end='\n\n')
    np.save(logits_pth, log_prob)

f.close()