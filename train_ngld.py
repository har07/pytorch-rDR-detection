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
from torch.optim import RMSprop, SGD, Adam
import torch.nn as nn
import torch.nn.functional as F
import torch
import datetime
import time
import inspect
import yaml
import timm

sys.path.insert(1, '../')
from lib.dataset import load_predefined_heldout_train_test
from lib.weights import get_class_weights, batch_samples_per_class
from sgld.sgld_optim import SGLD
from sgld.psgld_optim import pSGLD
from sgld.asgld_optim import ASGLD
from sgld.ksgld_optim import KSGLD
from sgld.eksgld_optim import EKSGLD
import lib.lr_setter as lr_setter

from torch.utils.tensorboard import SummaryWriter

print(f"Numpy version: {np.__version__}")
print(f"PyTorch version: {torch.__version__}")

# Various loading and saving constants.
default_save_model_path = "/content/model"
default_save_summaries_dir = "/content/logs"
default_yaml =  "config/train_eksgld.yaml"

parser = argparse.ArgumentParser(
                    description="Trains and saves neural network for "
                                "detection of diabetic retinopathy.")
parser.add_argument("-sm", "--save_model_path",
                    help="path to where graph model should be saved",
                    default=default_save_model_path)
parser.add_argument("-ss", "--save_summaries_dir",
                    help="path to folder where summaries should be saved",
                    default=default_save_summaries_dir)
parser.add_argument("-v", "--verbose",
                    help="print log per batch instead of per epoch",
                    default=False)
parser.add_argument("-y", "--yaml",
                    help="yaml config file location",
                    default=default_yaml)
parser.add_argument("-c", "--checkpoint", default="",
                    help="Checkpoint file")

args = parser.parse_args()
save_model_path = str(args.save_model_path)
save_summaries_dir = str(args.save_summaries_dir)
is_verbose = bool(args.verbose)
yaml_path = str(args.yaml)
checkpoint = str(args.checkpoint)

with open(yaml_path) as f:
    config = yaml.load(f, Loader=yaml.Loader)

model_type = config['model']
drop_rate = config['drop_rate']
seed = config['seed']
block_size = config['block_size']
block_decay = config['block_decay']

batch_size = config['dataset']['batch_size']
dataset_mean = config['dataset']['mean']
dataset_std = config['dataset']['std']
train_datadir = config['dataset']['train_dataset']
valid_datadir = config['dataset']['valid_dataset']
heldout_datadir = config['dataset']['heldout_dataset']

class_weight = config['class_weight']['method']
samples_per_class = config['class_weight']['samples_per_class']
class_weight_beta = config['class_weight']['beta']
per_minibatch = config['class_weight']['per_minibatch']

print("""
Saving model and graph checkpoints at: {},
Saving summaries at: {},
Training images folder: {},
Validation images folder: {},
Heldout images folder: {},
Optimizer config path: {}
""".format(save_model_path, save_summaries_dir, train_datadir, valid_datadir, heldout_datadir, yaml_path))

torch.cuda.set_device(0)
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

session_id_prefix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
f = open(f'{save_summaries_dir}/train_logs_{session_id_prefix}.txt', 'w')

optimizer_name = config['optimizer']
num_epochs = config['max_epoch']
limit_epoch = config['max_session_epoch']
if limit_epoch == 0:
    limit_epoch = num_epochs

train_dataset = None
val_dataset = None

_, val_dataset, train_dataset = load_predefined_heldout_train_test(heldout_datadir, valid_datadir, \
                                                    train_datadir, batch_size=batch_size, \
                                                    mean=dataset_mean, std=dataset_std)

# Base model InceptionV3 with global average pooling.
model = None
if model_type == 'resnet':
    model = torchvision.models.resnet101(pretrained=True, progress=True)
elif model_type == 'densenet':
    model = timm.create_model('densenet121', pretrained=True, num_classes=3)
else:
    model = timm.create_model('inception_v4', pretrained=True, num_classes=3, drop_rate=drop_rate)


# Reset the layer with the same amount of neurons as labels.

if model_type == 'resnet':
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)

model = model.cuda()

# lr decay following (Paradisa, 2022)
decay_by_loss = config[optimizer_name]['_decay_by_loss']
decay_rate = 0.
if decay_by_loss: 
    decay_rate = config[optimizer_name]['_decay_rate']


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

log_dir = f"{save_model_path}/runs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
writer = SummaryWriter(log_dir=log_dir)

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
            writer.writerow(['epoch', 'train_seconds',
                'train_loss', 'train_accuracy', 'accuracy', 'lr'])
        else:
            writer.writerow(["{}".format(x) for x in data[:6]] + ["{:0.4f}".format(x) for x in data[6:]])

def write_board(epoch, tloss, tacc, acc, lr):
    writer.add_scalar("Train Loss/train", tloss, epoch)
    writer.add_scalar("Train Accuracy/train", tacc, epoch)
    writer.add_scalar("Val Accuracy/train", acc, epoch)
    writer.add_scalar("Learning Rate", lr, epoch)

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
top20 = []

# Load checkpoint if provided
if checkpoint != "":
    chk = torch.load(checkpoint)
    start_epoch = chk['epoch'] + 1
    durations = chk['durations']
    step = chk['steps']
    top20 = chk['top20']
    optimizer.load_state_dict(chk['optimizer_state_dict'])
    model.load_state_dict(chk['model_state_dict'])

weights = get_class_weights(class_weight, len(samples_per_class), samples_per_class, class_weight_beta)
last_loss = 0.0
for epoch in range(start_epoch, limit_epoch+1):
    t0 = time.time()
    model.train()
    epoch_loss = 0.0
    epoch_acc = 0.0
    batch_num = 0
    accum_target = []
    for data, target in train_dataset:
        step += 1
        data = data.cuda()
        target = target.cuda()
        optimizer.zero_grad()
        output = model(data)
        output = F.log_softmax(output, dim=1)
        if epoch == 0:
            accum_target.extend(target.cpu().numpy())
        
        if per_minibatch:
            samples_per_class = batch_samples_per_class(len(samples_per_class), target)
            weights = get_class_weights(class_weight, len(samples_per_class), samples_per_class, class_weight_beta)

        loss = F.nll_loss(output, target, weight=torch.Tensor(weights).cuda())
        loss.backward()    # calc gradients
        
        # exception for SGD: do not perform lr decay
         # do not perform custom lr setting for built-in optimizer
        if optimizer_name in ['SGD', 'RMSprop']:
            optimizer.step()
        elif block_size > 0 and block_decay > 0:
            optimizer.step(lr=current_lr)

        epoch_loss += output.shape[0] * loss.item()

        prediction = output.data.max(1)[1]   # first column has actual prob.
        accuracy = np.mean(prediction.eq(target.data).cpu().numpy())*100

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
    val_accuracy, _ = lib.evaluation.evaluate(model, val_dataset)
    train_loss = np.mean(loss.item())
    train_acc = np.mean(accuracy)

    last_loss = train_loss
    if decay_by_loss and last_loss > train_loss:
        current_lr = current_lr * decay_rate

    print(f'Epoch: {epoch}\tTrain Sec: {elapsed:0.3f}')
    print(f'Epoch: {epoch}\tTLoss: {train_loss:0.3f}\tTAcc: {train_acc:0.3f}\tAcc: {val_accuracy:0.3f}')

    write_board(epoch, train_loss, train_acc, val_accuracy, current_lr)
    write_csv(session_id+".csv", data=[epoch, elapsed, train_loss, 
                                        train_acc, val_accuracy, current_lr])

    # Save the model weights max for the last 20 epochs
    # and top 10 val accuracy models
    save = False
    if len(top20) < 20 or top20[0] < val_accuracy or num_epochs - epoch < 20:
        save = True
        top20.append(val_accuracy)
        top20.sort(reverse=True)
        
    if save:
        torch.save({
                'model_state_dict': model.state_dict(),
                # 'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'steps': step,
                'lr': current_lr
            }, f"{save_model_path}/{session_id}_{epoch}.pt")

    # If current training session reach limit epoch, stop training:
    if limit_epoch > 0 and epoch >= limit_epoch:
        break

 # save params so that we can resume training

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'steps': step,
    'durations': durations,
    'top20': top20
}, f"{save_model_path}/{session_id}_chk.pt")

writer.flush()
print(f"epoch duration (mean +/- std): {np.mean(durations):.2f} +/- {np.std(durations):.2f}")
print(f"epoch duration (mean +/- std): {np.mean(durations):.2f} +/- {np.std(durations):.2f}", file=f)
f.close()