import numpy as np
import torch

def weight_inv_of_samples(count_classes, count_samples, power=1):
    weight = 1.0 / np.array(np.power(count_samples, power))
    weight = weight / np.sum(weight) * count_classes
    return weight

def weight_ess(count_classes, beta, count_samples):
    eff_num = 1.0 - np.power(beta, count_samples)
    weight = (1.0 - beta) / np.array(eff_num)
    weight = weight / np.sum(weight) * count_classes
    return weight

def weight_for_batch(method, count_classes, samples_per_class, batch_labels, beta = None):
    if method == 'ens':
        weight = weight_ess(count_classes, beta, samples_per_class)
    elif method == 'ins':
        weight = weight_inv_of_samples(count_classes, samples_per_class)
    elif method == 'isns':
        weight = weight_inv_of_samples(count_classes, samples_per_class, 0.5)
    else:
        return None

    batch_labels = batch_labels.to('cpu').numpy()
    weight = torch.tensor(weight).float()
    weight = weight.unsqueeze(0)
    weight = torch.tensor(np.array(weight.repeat(batch_labels.shape[0],1) * batch_labels))
    weight = weight.sum(1)
    weight = weight.unsqueeze(1)
    weight = weight.repeat(1, count_classes)
    return weight

# result:
# INS:  [0.07099362 0.22091163 2.70809475]
# ISNS:  [0.33556231 0.59193335 2.07250434]
# ENS (beta=0.9):  [1. 1. 1.]
# ENS (beta=0.99):  [0.99992133 0.99992133 1.00015733]
# ENS (beta=0.999):  [0.79548072 0.79551053 1.40900875]
# ENS (beta=0.9999):  [0.20671628 0.30993443 2.48334929]

# w1 = weight_inv_of_samples(3, [31699,10187,831])
# print("INS: ", w1)

# w2 = weight_inv_of_samples(3, [31699,10187,831], power=0.5)
# print("ISNS: ", w2)

# w3 = weight_ess(3, 0.9, [31699,10187,831])
# print("ENS (beta=0.9): ", w3)

# w4 = weight_ess(3, 0.99, [31699,10187,831])
# print("ENS (beta=0.99): ", w4)

# w5 = weight_ess(3, 0.999, [31699,10187,831])
# print("ENS (beta=0.999): ", w5)

# w6 = weight_ess(3, 0.9999, [31699,10187,831])
# print("ENS (beta=0.9999): ", w6)