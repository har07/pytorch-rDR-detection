import torch
import numpy as np
import torch.nn.functional as F
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
            output = F.log_softmax(output, dim=1)
            prediction = output.data.max(1)[1]   # first column has actual prob.
            val_accuracy = np.mean(prediction.eq(target.data).cpu().numpy())*100
            outputs.append(output)
            accuracies.append(val_accuracy)
        
    return np.mean(accuracies), output

def evaluate_nll(model, test_loader):
    model.eval()
    outputs = []
    accuracies = []
    loss_list = [] # list of per batch NLL loss
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.cuda()
            target = target.cuda()
            output = model(data)
            output = F.log_softmax(output, dim=1)

            loss = F.nll_loss(output, target)
            loss_list.append(loss.cpu().numpy())
        
    return np.mean(loss_list), output
