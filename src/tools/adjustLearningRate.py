import numpy as np


def adjust_learning_rate(epoch, opt, optimizer):
    '''reduce learning rate by opt.lr_decay_rate every time '''
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

