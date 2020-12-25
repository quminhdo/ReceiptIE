from torch.optim import Adam
from .radam import RAdam

def get_optimizer(opt, parameters):
    lr = opt.lr
    if opt.name == "Adam":
        optimizer = Adam(parameters, lr=lr)
    elif opt.name == "RAdam":
        optimizer = RAdam(parameters, lr=lr)
    else:
        raise Exception("Optimizer %s is not defined"%opt.name)
    return optimizer