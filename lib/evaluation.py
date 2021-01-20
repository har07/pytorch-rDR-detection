import torch
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, brier_score_loss
# from pytorch_lightning.metrics.functional import confusion_matrix

def _get_operations_by_names(graph, names):
    return [graph.get_operation_by_name(name) for name in names]


def _get_tensors_by_names(graph, names):
    return [graph.get_tensor_by_name(name) for name in names]

def evaluate(model, test_loader):
    model.eval()
    accum_target = []
    accum_pred = []
    # confusion matrix:
    # [[tn, fp]
    #  [fn, tp]]
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.cuda()
            target = target.cuda()
            output = model(data)
            output = output.reshape(-1)
            prediction = torch.round(torch.sigmoid(output))
            accum_pred.extend(prediction.cpu().numpy())
            accum_target.extend(target.cpu().numpy())
        
    # print('accum_pred: ', accum_pred)
    # print('accum_target: ', accum_target)    
    cf = confusion_matrix(accum_target, accum_pred)
    auc = roc_auc_score(accum_target, accum_pred)
    brier = brier_score_loss(accum_target, accum_pred)
    return cf, auc, brier
