import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def variable_length_mse(predict, target, length, gpu_id):   # predict's and target's shape (batch_size, max_len, input_size)

    length = torch.tensor(length, dtype=torch.float32).cuda(gpu_id)
    batch_size, _, _ = predict.shape

    distance_sum = (torch.sum((predict - target) ** 2, dim=(1, 2))).mul(1 / length)
    loss = torch.sum(distance_sum) / batch_size

    return loss


def variable_length_mse_truncated(predict, target, length, gpu_id, threshold_=0.1):   # predict's and target's shape (batch_size, max_len, input_size)

    length = torch.tensor(length, dtype=torch.float32).cuda(gpu_id)
    batch_size, _, _ = predict.shape

    dist = (predict - target)**2
    zero = torch.zeros_like(dist)

    truncated_dist = torch.where(dist<threshold_**2, zero, dist)
    # truncated_dist = torch.clamp((predict - target)**2, min=threshold_**2)

    distance_sum = (torch.sum(truncated_dist, dim=(1, 2))).mul(1 / length)
    loss = torch.sum(distance_sum) / batch_size

    return loss


def diff_loss(input1, input2):
    input1 = input1 - torch.mean(input1, dim=1, keepdim=True)
    input2 = input2 - torch.mean(input2, dim=1, keepdim=True)

    input1_l2_norm = torch.linalg.norm(input1, ord=2, dim=1, keepdim=True).detach()
    input2_l2_norm = torch.linalg.norm(input2, ord=2, dim=1, keepdim=True).detach()

    input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)
    input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

    diffLoss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))
    # diffLoss = torch.sum((input1_l2.t().mm(input2_l2)).pow(2))
    # diffLoss = torch.sum((input1_l2.t().mm(input2_l2)).pow(2)) / input1.shape[0]

    return diffLoss


def info_nce_loss(features, batch_size, gpu_id, n_views, temperature):
    '''
    From https://github.com/sthalles/SimCLR/blob/master/run.py
    '''

    labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.cuda(gpu_id)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda(gpu_id)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda(gpu_id)

    logits = logits / temperature
    return logits, labels
