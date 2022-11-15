'''
Referring to the code of MOCO, https://arxiv.org/abs/1911.05722
'''

import torch
import torch.nn as nn


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    """
    def __init__(self, base_encoder, input_size, hidden_units, layers, dropout, gpu_id,
                 K=65536, m=0.999, T=0.07, mlp=True, update_key_encoder=True, update_key_mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.gpu_id = gpu_id
        self.update_key_encoder = update_key_encoder
        self.update_key_mlp = update_key_mlp

        # create the encoders
        self.encoder_fc_q = base_encoder(input_size, hidden_units, layers, dropout)  # query encoder + mlp
        self.encoder_fc_k = base_encoder(input_size, hidden_units, layers, dropout)  # key encoder + mlp

        if mlp:   # hack: brute-force replacement
            self.encoder_fc_q.fc = nn.Sequential(nn.Linear(hidden_units*2, hidden_units*2),
                                                 nn.ReLU(),
                                                 nn.Linear(hidden_units*2, hidden_units*2))
            self.encoder_fc_k.fc = nn.Sequential(nn.Linear(hidden_units*2, hidden_units*2),
                                                 nn.ReLU(),
                                                 nn.Linear(hidden_units*2, hidden_units*2))

        for param in self.encoder_fc_k.parameters():
            param.requires_grad = False   # not update by gradient

        if self.update_key_encoder:
            for param_q, param_k in zip(self.encoder_fc_q.gru.parameters(), self.encoder_fc_k.gru.parameters()):
                param_k.data.copy_(param_q.data)  # initialize

        if self.update_key_mlp:
            for param_q, param_k in zip(self.encoder_fc_q.fc.parameters(), self.encoder_fc_k.fc.parameters()):
                param_k.data.copy_(param_q.data)  # initialize

        num_parameters = sum(param.numel() for param in self.encoder_fc_q.parameters())
        print('==> encoder_mlp parameters: {}. '.format(num_parameters))

        # create the queue
        self.register_buffer("queue", torch.randn(hidden_units*2, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder and key MLP
        """

        if self.update_key_encoder:
            for param_q, param_k in zip(self.encoder_fc_q.gru.parameters(), self.encoder_fc_k.gru.parameters()):
                param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

        if self.update_key_mlp:
            for param_q, param_k in zip(self.encoder_fc_q.fc.parameters(), self.encoder_fc_k.fc.parameters()):
                param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

        # if self.update_key_mlp:
        #     for param_q, param_k in zip(self.encoder_fc_q.fc.parameters(), self.encoder_fc_k.fc.parameters()):
        #         param_k.data = param_k.data * 0.999 + param_q.data * (1. - 0.999)

        # print('self.encoder_fc_k.gru.weight_ih_l0:\n', self.encoder_fc_k.gru.weight_ih_l0[:2, :2])
        # print('self.encoder_fc_k.fc[0].weight:\n', self.encoder_fc_k.fc[0].weight[:2, :2])
        # print()

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):

        batch_size = keys.shape[0]
        assert self.K % batch_size == 0  # for simplicity

        ptr = int(self.queue_ptr)
        self.queue[:, ptr:ptr + batch_size] = keys.T    # replace the keys at ptr (dequeue and enqueue)
        ptr = (ptr + batch_size) % self.K  # update pointer
        self.queue_ptr[0] = ptr

        # # if self.K < keys.shape[0]
        # batch_size = keys.shape[0] // 2
        # self.queue[:, :] = keys[:batch_size, :].T    # replace the keys at ptr (dequeue and enqueue)

    def forward(self, im_q, im_k, num_frame):
        """
        Input:
            im_q: a batch of query data
            im_k: a batch of key data
        Output:
            logits, targets
        """

        # compute query features
        features, _, q = self.encoder_fc_q(im_q, num_frame, self.gpu_id)   # queries: batch_size x hidden_unit*2
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            _, _, k = self.encoder_fc_k(im_k, num_frame, self.gpu_id)  # keys: batch_size x hidden_unit*2
            k = nn.functional.normalize(k, dim=1)


        # compute logits. Einstein sum is more intuitive
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)               # positive logits: batch_size x 1
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])  # negative logits: batch_size x 1

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda(self.gpu_id)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return features, logits, labels

