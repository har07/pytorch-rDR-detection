import numpy as np
import logging
import random
import sys
import torch
import torchvision
import torch.optim as optim
import torch.nn.functional as F
from lib.dataset import load_predefined_train_test
from sgld.sgld_optim import SGLD
import lib.lr_setter as lr_setter
import argparse
import datetime
import inspect
import yaml

import optuna

default_trial = 50
default_epochs = 10
default_batch_size = 128
default_seed = 0
default_yaml = "config/tuning_eyepacs.yaml"

parser = argparse.ArgumentParser(
                    description="Perform  hyperparameter tuning of SGLD optimizer for eyepacs classification.")
parser.add_argument("-s", "--study",
                    help="hyperparam optimization study name. determine database file to save to.")
parser.add_argument("-t", "--trials", help="number of trials to perform",
                    default=default_trial)
parser.add_argument("-e", "--epochs", help="number of epoch to perform",
                    default=default_epochs)
parser.add_argument("-o", "--optimizer",
                    help="optimizer name: sgld, sgld2, sgld3, psgld, psgld2, psgld3, asgld, ksgld")
parser.add_argument("-bs", "--blocksize", help="block decay size", default=0)
parser.add_argument("-bd", "--blockdecay", help="block decay", default=0)
parser.add_argument("-td", "--train_dataset",
                    help="path to folder that contains the train dataset")
parser.add_argument("-vd", "--valid_dataset",
                    help="path to folder that contains the validation dataset")
parser.add_argument("-y", "--yaml", help="yaml config file location",
                    default=default_yaml)

args = parser.parse_args()
if args.study:
    study_name = str(args.study)
else:
    study_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

yaml_path = str(args.yaml)
with open(yaml_path) as f:
    config = yaml.load(f, Loader=yaml.Loader)

seed = config['seed']
epochs = config['epoch']
trials = config['trials']
blocksize = config['block_size']
blockdecay = config['block_decay']
batch_size = config['batch_size']
train_dataset = config['train_dataset']
valid_dataset = config['valid_dataset']
optimizer_name = config['optimizer']

accept_model = False
if '_accept_model' in config[optimizer_name]:
    accept_model = config[optimizer_name]['_accept_model']

torch.cuda.set_device(0)
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

train_loader, _ = load_predefined_train_test(train_dataset, valid_dataset, bs=batch_size, valid_bs=batch_size)
model = torchvision.models.inception_v3(pretrained=True, progress=True, aux_logits=False)
model = model.cuda()

# use held-out training set for hyperparameter tuning
def train(trial, model, optimizer, train_loader, epochs, lr):
    current_lr = lr
    # check if optimizer.step has 'lr' param
    step_args = inspect.getfullargspec(optimizer.step)
    lr_param = 'lr' in step_args.args

    for epoch in range(1, epochs+1):
        model.train()
        epoch_loss = 0.0
        for data, target in train_loader:
            data = data.cuda()
            target = target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = F.binary_cross_entropy_with_logits(output, target, pos_weight=torch.Tensor([1.0]).cuda())
            loss.backward()    # calc gradients

            if blocksize > 0 and blockdecay > 0 and lr_param:
                optimizer.step(lr=current_lr)
            else:
                optimizer.step()

            epoch_loss += output.shape[0] * loss.item()
            trial.report(loss.item(), epoch)

            if trial.should_prune():
                raise optuna.TrialPruned()


        # update learning rate
        if blocksize > 0 and blockdecay > 0 and ((epoch) % blocksize) == 0:
            current_lr = current_lr * blockdecay
            if not lr_param:
                optimizer = lr_setter.update_lr(optimizer, current_lr)

        # epoch loss
        train_loss = epoch_loss / len(train_loader.dataset)

    return train_loss

def objective(trial):
    optim_params = {}
    fixed_params = config[optimizer_name]['fixed']
    for k in fixed_params:
        # skip parameter that start with "_"
        if k[0] == '_':
            continue
        v = fixed_params[k]
        if v or v == False:
            optim_params[k] = v
    tunable_params = config[optimizer_name]['tune']
    for k in tunable_params:
        tunable_param = tunable_params[k]
        suggest_type = next(iter(tunable_param))
        suggest_params = {}
        for l in tunable_params[suggest_type]:
            suggest_params[l] = tunable_params[suggest_type][l]
        suggests = eval(f"trial.suggest_{suggest_type}")(k, **suggest_params)
        optim_params[k] = suggests
        

    if accept_model:
        optimizer = eval(optimizer_name)(model, **optim_params)
    else:
        optimizer = eval(optimizer_name)(model.parameters(), **optim_params)

    lr = optim_params['lr']
    bce_loss = train(trial, model, optimizer, train_loader, epochs, lr)
    return bce_loss

def print_stats(study):
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

def main():
    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True, 
                        direction="minimize")
    study.optimize(objective, n_trials=trials)

    print_stats(study)

if __name__ == "__main__":
    main()