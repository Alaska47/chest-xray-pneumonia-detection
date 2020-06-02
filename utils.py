import torch

def pred_acc(original, predicted):
    return torch.div(predicted.eq(original).sum().float(), original.size()[0]).mean()
