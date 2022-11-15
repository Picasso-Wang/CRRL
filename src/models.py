import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.nn import Parameter


class LSTM_model(nn.Module):
    def __init__(self, in_dim=150, hidden_dim=150, n_layer=2, dropout_rate=0.2):
        super(LSTM_model, self).__init__()
        if n_layer > 1:
            self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden_dim, num_layers=n_layer,
                                batch_first=True, dropout=dropout_rate)
        else:
            self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden_dim, num_layers=n_layer,
                                batch_first=True)

    def forward(self, x):
        out, _ = self.lstm(x)
        return out


class Bi_LSTM(nn.Module):
    def __init__(self, in_dim=150, hidden_dim=150, n_layer=2, dropout_rate=0.2):
        super(Bi_LSTM, self).__init__()
        if n_layer > 1:
            self.biLSTM = nn.LSTM(input_size=in_dim, hidden_size=hidden_dim, num_layers=n_layer,
                                  batch_first=True, dropout=dropout_rate, bidirectional=True)
        else:
            self.biLSTM = nn.LSTM(input_size=in_dim, hidden_size=hidden_dim, num_layers=n_layer,
                                  batch_first=True, bidirectional=True)

    def forward(self, x):
        out, _ = self.biLSTM(x)   # out's shape (batch_size, video_len, 2*hidden_dim)
        return out


class GRU_model(nn.Module):
    def __init__(self, in_dim=150, hidden_dim=150, n_layer=2, dropout_rate=0.2):
        super(GRU_model, self).__init__()
        if n_layer > 1:
            self.gru = nn.GRU(input_size=in_dim, hidden_size=hidden_dim, num_layers=n_layer,
                              batch_first=True, dropout=dropout_rate)
        else:
            self.gru = nn.GRU(input_size=in_dim, hidden_size=hidden_dim, num_layers=n_layer,
                              batch_first=True)

    def forward(self, x):
        self.gru.flatten_parameters()

        out, _ = self.gru(x)
        return out


class Bi_GRU(nn.Module):
    def __init__(self, in_dim=150, hidden_dim=150, n_layer=2, dropout_rate=0.2):
        super(Bi_GRU, self).__init__()
        if n_layer > 1:
            self.gru = nn.GRU(input_size=in_dim, hidden_size=hidden_dim, num_layers=n_layer,
                              batch_first=True, dropout=dropout_rate, bidirectional=True)
        else:
            self.gru = nn.GRU(input_size=in_dim, hidden_size=hidden_dim, num_layers=n_layer,
                              batch_first=True, bidirectional=True)

    def forward(self, x):
        self.gru.flatten_parameters()

        out, _ = self.gru(x)
        return out



class Bi_GRU_packpad(nn.Module):
    def __init__(self, in_dim=150, hidden_dim=150, n_layer=2, dropout_rate=0.2, num_classes=60):
        super(Bi_GRU_packpad, self).__init__()
        if n_layer > 1:
            self.gru = nn.GRU(input_size=in_dim, hidden_size=hidden_dim, num_layers=n_layer,
                              batch_first=True, dropout=dropout_rate, bidirectional=True)
        else:
            self.gru = nn.GRU(input_size=in_dim, hidden_size=hidden_dim, num_layers=n_layer,
                              batch_first=True, bidirectional=True)

        self.fc = nn.Linear(hidden_dim*2, num_classes)

    def forward(self, x, n_frame, gpu_id):
        self.gru.flatten_parameters()   # put parameters to a single contiguous chunk of memory.

        length = torch.tensor(n_frame, dtype=torch.float32).cuda(gpu_id)
        length = torch.unsqueeze(length, dim=1)

        x_pack = rnn_utils.pack_padded_sequence(x, n_frame, batch_first=True)
        feature, _ = self.gru(x_pack)
        feature_pad, _ = rnn_utils.pad_packed_sequence(feature, batch_first=True)
        encoding = (1 / length).mul(torch.sum(feature_pad, dim=1))  # TAP: temporal average pooling. encoding's shape (batchsize, hidden_unit*2)
        encoding_fc = self.fc(encoding)

        return feature_pad, encoding, encoding_fc


class LinearClassifier_1layer(nn.Module):
    def __init__(self,  last_layer_dim=None, n_label=None):
        super(LinearClassifier_1layer, self).__init__()

        self.classifier_ = nn.Linear(last_layer_dim, n_label)
        self.initilize()

    def initilize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.fill_(0.0)

    def forward(self, x):
        return self.classifier_(x)


class LinearClassifier_2layer(nn.Module):
    def __init__(self,  last_layer_dim=None, n_label=None):
        super(LinearClassifier_2layer, self).__init__()

        self.classifier_ = nn.Sequential(
            nn.Linear(last_layer_dim, 2*last_layer_dim),
            nn.ReLU(),
            nn.Linear(2*last_layer_dim, n_label)
        )
        self.initilize()

    def initilize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.fill_(0.0)

    def forward(self, x):
        return self.classifier_(x)


class Nonlinear_2layer(nn.Module):
    def __init__(self,  last_layer_dim=None, out_dim=None):
        super(Nonlinear_2layer, self).__init__()

        self.classifier_ = nn.Sequential(
            nn.Linear(last_layer_dim, last_layer_dim),
            nn.ReLU(),
            nn.Linear(last_layer_dim, out_dim)
        )

    def forward(self, x):
        return self.classifier_(x)
