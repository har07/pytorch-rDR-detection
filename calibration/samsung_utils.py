import os
import sys
import pandas as pd
import time
import numpy as np
from scipy.special import logsumexp
sys.path.insert(1, '../')
from samsung_metrics import metrics_kfold

class Arguments:
  def __init__(self, dataset, model, method) -> None:
      self.dataset = dataset
      self.model = model
      self.method = method

class Logger:
    def __init__(self, base='./logs/'):
        self.res = []
        self.base = base
        os.makedirs(base, exist_ok=True)
        self.df = None

    def add(self, ns, metrics, args, info='', end='\n', silent=False):
        for m in metrics:
            self.res += [[args.dataset, args.model, args.method, ns, m, metrics[m], info]]
        if not silent:
            print('ns %s: acc %.4f, nll %.4f' % (ns, metrics['acc'], metrics['ll']), flush=True, end=end)

    def save(self, args, silent=True):
        self.df = pd.DataFrame(
            self.res, columns=['dataset', 'model', 'method', 'n_samples', 'metric', 'value', 'info'])
        dir = '%s-%s-%s.cvs' % (args.dataset, args.model, args.method)
        dir = os.path.join(self.base, dir)
        if not silent:
            print('Saved to:', dir, flush=True)
        self.df.to_csv(dir)

    def print(self):
        print(self.df, flush=True)

    def add_metrics_ts(self, ns, log_probs, targets, args, time_=0):

        if args.dataset == 'ImageNet':
            disable = ('misclass_MI_auroc', 'sce', 'ace')
            n_runs = 2
        else:
            n_runs = 5
            disable = ('misclass_MI_auroc', 'sce', 'ace', 'misclass_entropy_auroc@5', 'misclass_confidence_auroc@5')
        log_prob = logsumexp(np.dstack(log_probs), axis=2) - np.log(ns+1)
        metrics = metrics_kfold(log_prob, targets, n_splits=2, n_runs=n_runs, disable=disable)
        silent = (ns != 0 and (ns + 1) % 10 != 0)
        self.add(ns+1, metrics, args, silent=silent, end=' ')

        args.method = args.method + ' (ts)'
        metrics_ts = metrics_kfold(log_prob, targets, n_splits=2, n_runs=n_runs, temp_scale=True, disable=disable)
        self.add(ns+1, metrics_ts, args, silent=True)
        args.method = args.method[:-5]
        if not silent:
            print("time: %.3f" % (time.time() - time_))