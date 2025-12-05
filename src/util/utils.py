from torch import nn

from .misc import colorstr
import matplotlib.pyplot as plt

plt.switch_backend('agg')


def make_weights_for_balanced_classes(images, nclasses):
    '''
        Make a vector of weights for each image in the dataset, based
        on class frequency. The returned vector of weights can be used
        to create a WeightedRandomSampler for a DataLoader to have
        class balancing when sampling for a training batch.
            images - torchvisionDataset.imgs
            nclasses - len(torchvisionDataset.classes)
        https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3
    '''
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1  # item is (img-data, label-id)
    weight_per_class = [0.] * nclasses
    N = float(sum(count))  # total number of images
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]

    return weight


def separate_bn_paras(module):
    paras_only_bn = []
    paras_wo_bn = []

    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            paras_only_bn.extend(list(m.parameters()))
        else:
            paras_wo_bn.extend(
                [p for p in m.parameters(recurse=False)]  # only this module’s params
            )

    paras_only_bn = list({id(p): p for p in paras_only_bn}.values())
    paras_wo_bn = list({id(p): p for p in paras_wo_bn}.values())

    return paras_only_bn, paras_wo_bn


def warm_up_lr(batch, num_batch_warm_up, init_lr, optimizer):
    for params in optimizer.param_groups:
        params['lr'] = batch * init_lr / num_batch_warm_up


def schedule_lr(optimizer, factor=10.):
    for params in optimizer.param_groups:
        params['lr'] /= factor

    print(colorstr('magenta', optimizer))


def gen_plot(fpr, tpr):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlabel("FPR", fontsize=14)
    ax.set_ylabel("TPR", fontsize=14)
    ax.set_title("ROC Curve", fontsize=14)
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.plot(fpr, tpr, linewidth=2)
    #plt.show()
    plt.close(fig)
    return fig


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.sum = 0
        self.avg = 0
        self.val = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res
