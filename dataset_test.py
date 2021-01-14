from lib.dataset import load_split_train_test
from torch.autograd import Variable
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid

rootdir = '/home/hanif/Kuliah/exp/FGADR-Seg-set_Release/Seg-set-prep'
class_names = ['Non-rDR', 'rDR']

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

train_loader, test_loader = load_split_train_test(rootdir, bs=5)
# Get a batch of training data
inputs, classes = next(iter(train_loader))

# Make a grid from batch
out = make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])

input("Press Enter to continue...")

# for data, target in train_loader:
#     data, target = Variable(data), Variable(target)
#     # print('data[0]: ', data[0])
#     # print('target[1]: ', target[0])
#     # im = transforms.ToPILImage()(data[0]).convert("RGB")

#     # im.show()
#     break