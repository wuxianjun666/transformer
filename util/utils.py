import torch
import re

def load_record(path):
    f = open(path, 'r')
    losses = f.read()
    losses = re.sub('\\]', '', losses)
    losses = re.sub('\\[', '', losses)
    losses = re.sub('\\,', '', losses)
    losses = losses.split(' ')
    losses = [float(i) for i in losses]
    return losses, len(losses)


def load_weight(model):
    model.load_state_dict(torch.load("./saved/model-saved.pt"))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)