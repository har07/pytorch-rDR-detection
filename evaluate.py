import re
import os
import sys
import argparse
import random
import numpy as np
import lib.dataset
import lib.evaluation
import lib.metrics
import csv
from glob import glob
import torch
from lib.dataset import load_split_train_test
from lib.evaluation import evaluate

print(f"Numpy version: {np.__version__}")
print(f"PyTorch version: {torch.__version__}")

random.seed(432)

# Default settings.
default_fgadr_dir = "./data/fgadr/Seg-set-prep"
default_messidor2_dir = "./data/messidor2/Messidor-2-preproc"
default_load_model_path = "./tmp/model"
default_save_operating_thresholds_path = "./tmp/test_op_pts.csv"
default_batch_size = 32

parser = argparse.ArgumentParser(
                    description="Evaluate performance of trained graph "
                                "on test data set. "
                                "Specify --data_dir if you use the -o param.")
parser.add_argument("-f", "--fgadr", action="store_true",
                    help="evaluate performance on FGADR Segmentation Set")
parser.add_argument("-m2", "--messidor2", action="store_true",
                    help="evaluate performance on Messidor-2")
parser.add_argument("-o", "--other", action="store_true",
                    help="evaluate performance on your own dataset")
parser.add_argument("--data_dir", help="directory where data set resides")
parser.add_argument("-lm", "--load_model_path",
                    help="path to where graph model should be loaded from "
                         "creates an ensemble if paths are comma separated "
                         "or a regexp",
                    default=default_load_model_path)
parser.add_argument("-so", "--save_operating_thresholds_path",
                    help="path to where operating points metrics should be saved",
                    default=default_save_operating_thresholds_path)
parser.add_argument("-b", "--batch_size",
                    help="batch size", default=default_batch_size)
parser.add_argument("-op", "--operating_threshold",
                    help="operating threshold", default=0.5)

args = parser.parse_args()

if bool(args.fgadr) == bool(args.messidor2) == bool(args.other):
    print("Can only evaluate one data set at once!")
    parser.print_help()
    sys.exit(2)

if args.data_dir is not None:
    data_dir = str(args.data_dir)
elif args.fgadr:
    data_dir = default_fgadr_dir
elif args.messidor2:
    data_dir = default_messidor2_dir
elif args.other and args.data_dir is None:
    print("Please specify --data_dir.")
    parser.print_help()
    sys.exit(2)

load_model_path = str(args.load_model_path)
batch_size = int(args.batch_size)
save_operating_thresholds_path = str(args.save_operating_thresholds_path)
operating_threshold = float(args.operating_threshold)

print("""
Evaluating: {},
Saving operating thresholds metrics at: {},
Using operating treshold: {},
""".format(data_dir, save_operating_thresholds_path, operating_threshold))
print("Trying to load model(s):\n{}".format("\n".join(load_model_paths)))

# Other setting variables.
num_channels = 3
num_workers = 8
prefetch_buffer_size = 2 * batch_size
num_thresholds = 200
kepsilon = 1e-7

# Define thresholds.
thresholds = lib.metrics.generate_thresholds(num_thresholds, kepsilon) \
                + [operating_threshold]

got_all_y = False
all_y = []

all_predictions = []

# Base model InceptionV3 with global average pooling.
model = torchvision.models.inception_v3(pretrained=True, progress=True, aux_logits=False)

# Reset the layer with the same amount of neurons as labels.
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)
model.load_state_dict(torch.load(load_model_path))
model = model.cuda()

_, val_dataset = load_split_train_test(train_dir, valid_size=1.0)

cf, auc, brier = evaluate(model, val_dataset)
tn, fp, fn, tp = cf.ravel()
val_itmes = tn+fp+fn+tp
val_accuracy = (tn + tp)/val_itmes
val_sensitivity = tp/(tp + fn)
val_specificity = tn/(tn + fp)

# Print total roc auc score for validation.
print(f"Brier score: {brier:6.4}, AUC: {auc:10.8}")

# Print confusion matrix.
print(f"Confusion matrix at operating threshold {operating_threshold:0.3f}")
print(cf)

# Print sensitivity and specificity.
print("Specificity: {0:0.4f}, Sensitivity: {1:0.4f}" \
        .format(val_specificity, val_sensitivity))

# Write sensitivities and specificities to file.
with open(save_operating_thresholds_path, 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ')
    writer.writerow(['accuracy', 'specificity', 'sensitivity'])

    for idx in range(num_thresholds):
        writer.writerow([
            "{:0.4f}".format(x) for x in [
                thresholds[idx], test_specificities[idx],
                test_sensitivities[idx]]])

sys.exit(0)