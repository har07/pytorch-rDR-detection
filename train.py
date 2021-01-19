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
from lib.dataset import load_split_train_test
from torch.optim import RMSprop
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import datetime

print(f"Numpy version: {np.__version__}")
print(f"PyTorch version: {torch.__version__}")

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
random.seed(432)

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

args = parser.parse_args()
dataset_dir = str(args.dataset_dir)
save_model_path = str(args.save_model_path)
save_summaries_dir = str(args.save_summaries_dir)
is_verbose = bool(args.verbose)

print("""
Dataset images folder: {},
Saving model and graph checkpoints at: {},
Saving summaries at: {},
""".format(dataset_dir, save_model_path, save_summaries_dir))

# Various constants.
num_channels = 3
num_workers = 8
# normalization_fn = rescale_min_1_to_1

# Hyper-parameters for training.
learning_rate = 1e-3
decay = 4e-5
train_batch_size = 32

# Hyper-parameters for validation.
min_epochs = 50
num_epochs = 200
wait_epochs = 10
min_delta_auc = 0.01
val_batch_size = 32
num_thresholds = 200
kepsilon = 1e-7

# Define thresholds.
thresholds = lib.metrics.generate_thresholds(num_thresholds, kepsilon) + [0.5]

train_dataset, val_dataset = load_split_train_test(dataset_dir)

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
            writer.writerow(['epoch', 'count_data', 'tn', 'fp', 'fn', 'tp', 'train_loss', 'train_accuracy', 'accuracy', \
                'sensitivity', 'specificity', 'auc', 'brier'])
        else:
            writer.writerow(["{}".format(x) for x in data[:6]] + ["{:0.4f}".format(x) for x in data[6:]])

session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
write_csv(session_id+".csv", header=True)

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    epoch_acc = 0.0
    batch_num = 0
    for data, target in train_dataset:
        data, target = Variable(data), Variable(target)
        data = data.cuda()
        target = target.cuda()

        output = model(data)
        target = target.unsqueeze(1)
        target = target.float()
        loss = F.binary_cross_entropy_with_logits(output, target)
        loss.backward()    # calc gradients
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

    # Perform validation.
    cf, auc, brier = lib.evaluation.evaluate(model, train_dataset)
    tn, fp, fn, tp = cf.ravel()
    val_itmes = tn+fp+fn+tp
    val_accuracy = ((tn + tp)/val_itmes)*100
    val_sensitivity = tp/(tp + fn)
    val_specificity = tn/(tn + fp)
    val_auc = auc
    train_loss = epoch_loss / len(train_dataset.dataset)
    train_acc = epoch_acc/ len(train_dataset.dataset)

    print('Epoch: {}\tCount Data: {}\tTN: {}\tFP: {}\tFN: {}\tTP:{}'.format(epoch, val_itmes, tn, fp, fn, tp))
    print('TLoss: {:0.3f}\tTAcc: {:0.3f}\tAcc: {:0.3f}\tSn: {:0.3f}\tSp: {:0.3f}\tAUC: {:10.8}\tBrier: {:8.6}'
            .format(train_loss, train_acc, val_accuracy, val_sensitivity,
                      val_specificity, val_auc, brier))

    write_csv(session_id+".csv", data=[epoch, val_itmes, tn, fp, fn, tp, train_loss, 
                                        train_acc, val_accuracy, val_sensitivity, 
                                        val_specificity, val_auc, brier])

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
        torch.save(model.state_dict(), save_model_path + "/" + session_id+".pt")

        # Reset waited epochs.
        waited_epochs = 0