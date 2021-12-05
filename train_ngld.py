import numpy as np
import torchvision
import os
import random
import sys
import argparse
import csv
from glob import glob
import lib.metrics
import lib.dataset
import lib.evaluation
import lib.lr_setter as lr_setter
# from lib.preprocess import rescale_min_1_to_1, rescale_0_to_1
from lib.dataset import load_split_train_test, load_predefined_train_test
from sgld.asgld_optim import ASGLD
from sgld.psgld_optim import pSGLD
from sgld.ksgld_optim import KSGLD
from sgld.eksgld_optim import EKSGLD
from torch.optim import RMSprop, SGD
import torch.nn as nn
import torch.nn.functional as F
import torch
import datetime
import time
import inspect
import yaml
from torch.utils.tensorboard import SummaryWriter

print(f"Numpy version: {np.__version__}")
print(f"PyTorch version: {torch.__version__}")

random.seed(432)

# Various loading and saving constants.
default_dataset_dir = "./data/fgadr/Seg-set-prep"
default_save_model_path = "/content/model"
default_save_summaries_dir = "/content/logs"
default_save_operating_thresholds_path = "./tmp/op_pts.csv"
default_yaml =  "config/train_psgld.yaml"

parser = argparse.ArgumentParser(
                    description="Trains and saves neural network for "
                                "detection of diabetic retinopathy.")
parser.add_argument("-d", "--dataset_dir",
                    help="path to folder that contains the dataset",
                    default=default_dataset_dir)
parser.add_argument("-sm", "--save_model_path",
                    help="path to where graph model should be saved",
                    default=default_save_model_path)
parser.add_argument("-ss", "--save_summaries_dir",
                    help="path to folder where summaries should be saved",
                    default=default_save_summaries_dir)
parser.add_argument("-v", "--verbose",
                    help="print log per batch instead of per epoch",
                    default=False)
parser.add_argument("-b", "--balance",
                    help="resample dataset to balance per class data",
                    default=False)
parser.add_argument("-td", "--train_dataset",
                    help="path to folder that contains the train dataset")
parser.add_argument("-vd", "--valid_dataset",
                    help="path to folder that contains the validation dataset")
parser.add_argument("-y", "--yaml",
                    help="yaml config file location",
                    default=default_yaml)

args = parser.parse_args()
dataset_dir = str(args.dataset_dir)
save_model_path = str(args.save_model_path)
save_summaries_dir = str(args.save_summaries_dir)
is_verbose = bool(args.verbose)
balance = bool(args.balance)
train_dataset = str(args.train_dataset)
valid_dataset = str(args.valid_dataset)
yaml_path = str(args.yaml)

print("""
Dataset images folder: {},
Saving model and graph checkpoints at: {},
Saving summaries at: {},
Training images folder: {},
Validation images folder: {},
Optimizer config path: {}
""".format(dataset_dir, save_model_path, save_summaries_dir, train_dataset, valid_dataset, yaml_path))

with open(yaml_path) as f:
    config = yaml.load(f, Loader=yaml.Loader)

seed = config['seed']
block_size = config['block_size']
block_decay = config['block_decay']

dataset_params = config['dataset']
train_batch = dataset_params['train_batch']
test_batch = dataset_params['test_batch']

torch.cuda.set_device(0)
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

session_id_prefix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
f = open(f'{save_summaries_dir}/train_logs_{session_id_prefix}.txt', 'w')
optimizer_name = config['optimizer']

# Hyper-parameters for validation.
min_epochs = config['min_epochs']
num_epochs = config['epoch']
wait_epochs = config['wait_epoch']
min_delta_auc = 0.01


if train_dataset != 'None' and valid_dataset != 'None':
    train_dataset, val_dataset = load_predefined_train_test(train_dataset, valid_dataset, bs=train_batch, valid_bs=test_batch)
else:
    train_dataset, val_dataset = load_split_train_test(dataset_dir, bs=train_batch, valid_bs=test_batch, valid_size=0.2, balanced=balance)

# Base model InceptionV3 with global average pooling.
model = torchvision.models.inception_v3(pretrained=False, progress=True, aux_logits=False)

# Reset the layer with the same amount of neurons as labels.
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)
model = model.cuda()

# Define optimizer.
accept_model = False
if '_accept_model' in config[optimizer_name]:
    accept_model = config[optimizer_name]['_accept_model']
optim_params = {}
if optimizer_name in config:
    optim_params2 = config[optimizer_name]
    for k in optim_params2:
        # skip parameter that start with "_"
        if k[0] == '_':
            continue
        v = optim_params2[k]
        if v or v == False:
            optim_params[k] = v

session_id = f"{optimizer_name}_{session_id_prefix}"
print('optimizer: ', optimizer_name)
print('optimizer params: ', optim_params)
print('optimizer: ', optimizer_name, file=f)
print('optimizer params: ', optim_params, file=f)
if accept_model:
    optimizer = eval(optimizer_name)(model, **optim_params)
else:
    optimizer = eval(optimizer_name)(model.parameters(), **optim_params)

writer = SummaryWriter(log_dir=f"runs/{session_id}")

# Train for the specified amount of epochs.
latest_peak_auc = 0
waited_epochs = 0

# writer = SummaryWriter()

def print_training_status(epoch, num_epochs, batch_num, xent, i_step=None):
    def length(x): return len(str(x))

    m = []
    m.append(
        f"Epoch: {{0:>{length(num_epochs)}}}/{{1:>{length(num_epochs)}}}"
        .format(epoch, num_epochs))
    m.append(f"Batch: {batch_num:>4}, Loss: {xent:6.4}")

    if i_step is not None:
        m.append(f"Step: {i_step:>10}")

    print(", ".join(m))

def write_csv(filename, header=False, data=[]):
    mode = 'a'
    if header:
        mode = 'w'
    with open(save_summaries_dir + '/' + filename, mode) as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        if header:
            writer.writerow(['epoch', 'count_data', 'train_seconds', 'tn', 'fp', 'fn', 'tp', 
                'train_loss', 'train_accuracy', 'accuracy', 
                'sensitivity', 'specificity', 'auc', 'brier'])
        else:
            writer.writerow(["{}".format(x) for x in data[:6]] + ["{:0.4f}".format(x) for x in data[6:]])

def write_board(epoch, tloss, tacc, acc, sn, sp, auc, brier):
    writer.add_scalar("Train Loss/train", tloss, epoch)
    writer.add_scalar("Train Accuracy/train", tacc, epoch)
    writer.add_scalar("Val Accuracy/train", acc, epoch)
    writer.add_scalar("Sensitivity/train", sn, epoch)
    writer.add_scalar("Specificity/train", sp, epoch)
    writer.add_scalar("AUC/train", auc, epoch)
    writer.add_scalar("Brier/train", brier, epoch)

session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
write_csv(session_id+".csv", header=True)

step = 0
current_lr = optim_params["lr"]

# check if optimizer.step has 'lr' param
step_args = inspect.signature(optimizer.step)
lr_param = 'lr' in step_args.parameters

val_accuracy=0

start_epoch = 1
durations = []

for epoch in range(num_epochs):
    t0 = time.time()
    model.train()
    epoch_loss = 0.0
    epoch_acc = 0.0
    batch_num = 0
    accum_target = []
    for data, target in train_dataset:
        data = data.cuda()
        target = target.cuda()
        optimizer.zero_grad()
        output = model(data)
        target = target.unsqueeze(1)
        target = target.float()
        if epoch == 0:
            accum_target.extend(target.cpu().numpy())
        loss = F.binary_cross_entropy_with_logits(output, target, pos_weight=torch.Tensor([4.0]).cuda())
        loss.backward()    # calc gradients
        
        # exception for SGD: do not perform lr decay
        if optimizer_name == 'optim.SGD':
            optimizer.step()
        elif block_size > 0 and block_decay > 0 and lr_param:
            optimizer.step(lr=current_lr)
        else:
            optimizer.step()

        epoch_loss += output.shape[0] * loss.item()

        prediction = torch.round(torch.sigmoid(output))
        accuracy = np.mean(prediction.eq(target.data).cpu().numpy())*100
        epoch_acc += output.shape[0] *accuracy

        # Print a nice training status. 
        if is_verbose:
            print_training_status(
                epoch, num_epochs, batch_num, loss)
        batch_num += 1

    # measure training time
    elapsed = time.time() - t0
    durations.append(elapsed)

    # update learning rate for next epoch
    if block_size > 0 and block_decay > 0 and ((epoch) % block_size) == 0:
        current_lr = current_lr * block_decay
        if not lr_param:
            optimizer = lr_setter.update_lr(optimizer, current_lr)

    # inspect training data composition in first epoch
    eval_verbose = False
    if epoch == 0:
        eval_verbose = True
        class_0 = len([x for x in accum_target if int(x) == 0])
        print('training composition: 0={}, 1={}'.format(class_0, len(accum_target)-class_0))

    # Perform validation.
    cf, auc, brier = lib.evaluation.evaluate(model, val_dataset, verbose=eval_verbose)
    tn, fp, fn, tp = cf.ravel()
    val_itmes = tn+fp+fn+tp
    val_accuracy = ((tn + tp)/val_itmes)*100
    val_sensitivity = tp/(tp + fn)
    val_specificity = tn/(tn + fp)
    val_auc = auc
    train_loss = epoch_loss / len(train_dataset.dataset)
    train_acc = epoch_acc/ len(train_dataset.dataset)

    print(f'Epoch: {epoch}\tCount Data: {val_itmes}\tTrain Sec: {elapsed:0.3f}' + 
            f'\tTN: {tn}\tFP: {fp}\tFN: {fn}\tTP:{tp}')
    print(f'TLoss: {train_loss:0.3f}\tTAcc: {train_acc:0.3f}\tAcc: {val_accuracy:0.3f}\tSn: {val_sensitivity:0.3f}' + 
            f'\tSp: {val_specificity:0.3f}\tAUC: {val_auc:10.8}\tBrier: {brier:8.6}')

    write_board(epoch, train_loss, train_acc, val_accuracy, val_sensitivity, val_specificity, val_auc, brier)
    write_csv(session_id+".csv", data=[epoch, val_itmes, elapsed, tn, fp, fn, tp, train_loss, 
                                        train_acc, val_accuracy, val_sensitivity, 
                                        val_specificity, val_auc, brier])

    # Save the model weights max for the last 20 epochs
    if num_epochs - epoch < 20:
        torch.save({
                'model_state_dict': model.state_dict(),
                # 'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'steps': step,
                'lr': current_lr
            }, f"{save_model_path}/{session_id}_{epoch}.pt")

    if val_auc < latest_peak_auc + min_delta_auc:
        # Stop early if peak of val auc has been reached.
        # If it is lower than the previous auc value, wait up to `wait_epochs`
        #  to see if it does not increase again.
        if epoch < min_epochs:
            continue

        if wait_epochs == waited_epochs:
            print("Stopped early at epoch {0} with saved peak auc {1:10.8}"
                .format(epoch+1, latest_peak_auc))
            break

        waited_epochs += 1
    else:
        latest_peak_auc = val_auc
        print(f"New peak auc reached: {val_auc:10.8}")

        # Save the model weights.
        torch.save({
            'model_state_dict': model.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'steps': step,
            'lr': current_lr
        }, f"{save_model_path}/{session_id}_best.pt")

        # Reset waited epochs.
        waited_epochs = 0

writer.flush()
print(f"epoch duration (mean +/- std): {np.mean(durations):.2f} +/- {np.std(durations):.2f}")
print(f"epoch duration (mean +/- std): {np.mean(durations):.2f} +/- {np.std(durations):.2f}", file=f)
f.close()