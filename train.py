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
# from lib.preprocess import rescale_min_1_to_1, rescale_0_to_1
from lib.dataset import load_split_train_test, load_predefined_train_test
from sgld.asgld_optim import ASGLD
from sgld.kfac_precond import KFAC
from sgld.sgld_optim import SGLD, pSGLD
from torch.optim import RMSprop, SGD
import torch.nn as nn
import torch.nn.functional as F
import torch
import datetime
import time
from torch.utils.tensorboard import SummaryWriter

print(f"Numpy version: {np.__version__}")
print(f"PyTorch version: {torch.__version__}")

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Various loading and saving constants.
default_dataset_dir = "./data/fgadr/Seg-set-prep"
default_save_model_path = "/content/model"
default_save_summaries_dir = "/content/logs"
default_save_operating_thresholds_path = "./tmp/op_pts.csv"

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
parser.add_argument("-pw", "--positive_weight",
                    help="weight factor for postive class",
                    default=4.0)
parser.add_argument("-bs", "--batch_size",
                    help="batch size",
                    default=32)
parser.add_argument("-me", "--max_epoch",
                    help="number of max training epoch",
                    default=200)
parser.add_argument("-we", "--wait_epoch",
                    help="number of epoch before terminating training if AUC doesn't increase",
                    default=10)
parser.add_argument("-c", "--checkpoint", default="",
                    help="Checkpoint file")
parser.add_argument("-sd", "--seed", default=432,
                    help="Fix random seed for reproducability")

args = parser.parse_args()
dataset_dir = str(args.dataset_dir)
save_model_path = str(args.save_model_path)
save_summaries_dir = str(args.save_summaries_dir)
is_verbose = bool(args.verbose)
balance = bool(args.balance)
train_dataset = str(args.train_dataset)
valid_dataset = str(args.valid_dataset)
positive_weight = float(args.positive_weight)
batch_size = int(args.batch_size)
max_epoch = int(args.max_epoch)
wait_epochs = int(args.wait_epoch)
checkpoint = str(args.checkpoint)
seed = int(args.seed)

torch.cuda.set_device(0)
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

print("""
Dataset images folder: {},
Saving model and graph checkpoints at: {},
Saving summaries at: {},
Training images folder: {},
Validation images folder: {},
""".format(dataset_dir, save_model_path, save_summaries_dir, train_dataset, valid_dataset))

# Various constants.
num_channels = 3
num_workers = 8
# normalization_fn = rescale_min_1_to_1

# Hyper-parameters for training.
learning_rate = 1e-3
decay = 4e-5

# Hyper-parameters for validation.
min_epochs = 0
min_delta_auc = 0.01
val_batch_size = 32
num_thresholds = 200
kepsilon = 1e-7

# Define thresholds.
thresholds = lib.metrics.generate_thresholds(num_thresholds, kepsilon) + [0.5]

if train_dataset != 'None' and valid_dataset != 'None':
    train_dataset, val_dataset = load_predefined_train_test(train_dataset, valid_dataset, bs=batch_size, valid_bs=batch_size)
else:
    train_dataset, val_dataset = load_split_train_test(dataset_dir, bs=batch_size, valid_bs=batch_size, valid_size=0.2, balanced=balance)

# Base model InceptionV3 with global average pooling.
model = torchvision.models.inception_v3(pretrained=True, progress=True, aux_logits=False)

# Reset the layer with the same amount of neurons as labels.
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)
model = model.cuda()

# Define optimizer.
optimizer = RMSprop(model.parameters(), lr=learning_rate, weight_decay=decay)

# Train for the specified amount of epochs.
# Can be stopped early if peak of validation auc (Area under curve)
#  is reached.
latest_peak_auc = 0
waited_epochs = 0

# Load checkpoint if provided
start_epoch = 0
if checkpoint != "":
    chk = torch.load(checkpoint)
    start_epoch = chk['epoch'] + 1
    latest_peak_auc = chk['latest_peak_auc']
    waited_epochs = chk['waited_epochs']
    optimizer.load_state_dict(chk['optimizer_state_dict'])
    model.load_state_dict(chk['model_state_dict'])

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

for epoch in range(start_epoch, max_epoch):
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
        loss = F.binary_cross_entropy_with_logits(output, target, pos_weight=torch.Tensor([positive_weight]).cuda())
        loss.backward()    # calc gradients
        optimizer.step()

        epoch_loss += output.shape[0] * loss.item()

        prediction = torch.round(torch.sigmoid(output))
        accuracy = np.mean(prediction.eq(target.data).cpu().numpy())*100
        epoch_acc += output.shape[0] *accuracy

        # Print a nice training status. 
        if is_verbose:
            print_training_status(
                epoch, max_epoch, batch_num, loss)
        batch_num += 1

    # measure training time
    elapsed = time.time() - t0

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

    print('Epoch: {}\tCount Data: {}\tTrain Sec: {:0.3f}\tTN: {}\tFP: {}\tFN: {}\tTP:{}'
            .format(epoch, val_itmes, elapsed, tn, fp, fn, tp))
    print('TLoss: {:0.3f}\tTAcc: {:0.3f}\tAcc: {:0.3f}\tSn: {:0.3f}\tSp: {:0.3f}\tAUC: {:10.8}\tBrier: {:8.6}'
            .format(train_loss, train_acc, val_accuracy, val_sensitivity,
                      val_specificity, val_auc, brier))

    write_board(epoch, train_loss, train_acc, val_accuracy, val_sensitivity, val_specificity, val_auc, brier)
    write_csv(session_id+".csv", data=[epoch, val_itmes, elapsed, tn, fp, fn, tp, train_loss, 
                                        train_acc, val_accuracy, val_sensitivity, 
                                        val_specificity, val_auc, brier])

    writer.flush()

    last_epoch = epoch == max_epoch-1
    if val_auc < latest_peak_auc + min_delta_auc:
        # Stop early if peak of val auc has been reached.
        # If it is lower than the previous auc value, wait up to `wait_epochs`
        #  to see if it does not increase again.
        if epoch < min_epochs:
            continue

        if wait_epochs == waited_epochs:
            last_epoch = True
            print("Stopped early at epoch {0} with saved peak auc {1:10.8}"
                .format(epoch+1, latest_peak_auc))

        waited_epochs += 1
    else:
        latest_peak_auc = val_auc
        print(f"New peak auc reached: {val_auc:10.8}")

        # Save the model weights.
        torch.save(model.state_dict(), f"{save_model_path}/{session_id}_epoch{epoch}.pt")

        # Reset waited epochs.
        waited_epochs = 0

    if last_epoch:
        # save params so that we can resume training
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'latest_peak_auc': latest_peak_auc,
            'waited_epochs': waited_epochs,
        }, f"{save_model_path}/{session_id}_chk.pt")
        break