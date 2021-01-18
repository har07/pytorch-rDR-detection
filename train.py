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

print(f"Numpy version: {np.__version__}")
print(f"PyTorch version: {torch.__version__}")

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
random.seed(432)

# Various loading and saving constants.
default_train_dir = "./data/eyepacs/bin2/train"
default_val_dir = "./data/eyepacs/bin2/validation"
default_save_model_path = "./tmp/model"
default_save_summaries_dir = "./tmp/logs"
default_save_operating_thresholds_path = "./tmp/op_pts.csv"

parser = argparse.ArgumentParser(
                    description="Trains and saves neural network for "
                                "detection of diabetic retinopathy.")
parser.add_argument("-t", "--train_dir",
                    help="path to folder that contains training tfrecords",
                    default=default_train_dir)
parser.add_argument("-v", "--val_dir",
                    help="path to folder that contains validation tfrecords",
                    default=default_val_dir)
parser.add_argument("-sm", "--save_model_path",
                    help="path to where graph model should be saved",
                    default=default_save_model_path)
parser.add_argument("-ss", "--save_summaries_dir",
                    help="path to folder where summaries should be saved",
                    default=default_save_summaries_dir)
parser.add_argument("-so", "--save_operating_thresholds_path",
                    help="path to where operating points should be saved",
                    default=default_save_operating_thresholds_path)

args = parser.parse_args()
train_dir = str(args.train_dir)
val_dir = str(args.val_dir)
save_model_path = str(args.save_model_path)
save_summaries_dir = str(args.save_summaries_dir)
save_operating_thresholds_path = str(args.save_operating_thresholds_path)

print("""
Training images folder: {},
Validation images folder: {},
Saving model and graph checkpoints at: {},
Saving summaries at: {},
Saving operating points at: {},
""".format(train_dir, val_dir, save_model_path, save_summaries_dir,
           save_operating_thresholds_path))

# Various constants.
num_channels = 3
num_workers = 8
# normalization_fn = rescale_min_1_to_1

# Hyper-parameters for training.
learning_rate = 1e-3
decay = 4e-5
train_batch_size = 32

# Hyper-parameters for validation.
num_epochs = 200
wait_epochs = 10
min_delta_auc = 0.01
val_batch_size = 32
num_thresholds = 200
kepsilon = 1e-7

# Define thresholds.
thresholds = lib.metrics.generate_thresholds(num_thresholds, kepsilon) + [0.5]

train_dataset, val_dataset = load_split_train_test(train_dir)

# Base model InceptionV3 with global average pooling.
model = torchvision.models.inception_v3(pretrained=True, progress=True)

# Reset the layer with the same amount of neurons as labels.
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)
model = model.cuda()

# Define optimizer.
optimizer = RMSprop(model.parameters(), learning_rate=learning_rate, weight_decay=decay)

def print_training_status(epoch, num_epochs, batch_num, xent, i_step=None):
    def length(x): return len(str(x))

    m = []
    m.append(
        f"Epoch: {{0:>{length(num_epochs)}}}/{{1:>{length(num_epochs)}}}"
        .format(epoch, num_epochs))
    m.append(f"Batch: {batch_num:>4}, Xent: {xent:6.4}")

    if i_step is not None:
        m.append(f"Step: {i_step:>10}")

    print(", ".join(m), end="\r")

for epoch in range(num_epochs):
    model.train()
    batch_num = 0
    for data, target in train_dataset:
        data, target = Variable(data), Variable(target)
        data = data.cuda()
        target = target.cuda()

        output = model(data)
        # NOTES: how to implement the equivalent of tf.nn.sigmoid_cross_entropy_with_logits ?
        # someone said we can use binary_crossentropy, but we need make the NN to return
        # sigmoid activation function
        # https://discuss.pytorch.org/t/equivalent-of-tensorflows-sigmoid-cross-entropy-with-logits-in-pytorch/1985/10
        # concluded that this is equivalent to tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits())
        loss = F.binary_cross_entropy(output, target)
        loss.backward()    # calc gradients

        # Print a nice training status. 
        # NOTE: not sure what is equivalent to global_step in pytorch, so we just removed global_step
        print_training_status(
            epoch, num_epochs, batch_num, loss)
        batch_num += 1

    # TODO: # Retrieve training brier score.
    # train_brier = sess.run(brier)
    # print("\nEnd of epoch {0}! (Brier: {1:8.6})".format(epoch, train_brier))

    # TODO: # Perform validation.
    # val_auc = lib.evaluation.perform_test(
    #     sess=sess, init_op=val_init_op,
    #     summary_writer=train_writer, epoch=epoch)

    # val_auc = 0
    # if val_auc < latest_peak_auc + min_delta_auc:
    #     # Stop early if peak of val auc has been reached.
    #     # If it is lower than the previous auc value, wait up to `wait_epochs`
    #     #  to see if it does not increase again.

    #     if wait_epochs == waited_epochs:
    #         print("Stopped early at epoch {0} with saved peak auc {1:10.8}"
    #             .format(epoch+1, latest_peak_auc))
    #         break

    #     waited_epochs += 1
    # else:
    #     latest_peak_auc = val_auc
    #     print(f"New peak auc reached: {val_auc:10.8}")

    #     # Save the model weights.
    #     torch.save(model.state_dict(), save_model_path)

    #     # Reset waited epochs.
    #     waited_epochs = 0