import torch
import numpy as np
import auc_mu
from sklearn.metrics import confusion_matrix, roc_auc_score, brier_score_loss
# from pytorch_lightning.metrics.functional import confusion_matrix

def brier_multi(targets, probs):
  return np.mean(np.sum((probs - targets)**2, axis=1))

def evaluate(model, test_loader):
    model.eval()
    outputs = []
    accuracies = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.cuda()
            target = target.cuda()
            output = model(data)
            prediction = output.data.max(1)[1]   # first column has actual prob.
            val_accuracy = np.mean(prediction.eq(target.data).cpu().numpy())*100
            outputs.append(output)
            accuracies.append(val_accuracy)
        
    return np.mean(accuracies), output
