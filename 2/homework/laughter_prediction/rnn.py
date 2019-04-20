import torch
import torch.nn as nn
import torch.nn.functional as F


class LibrosaFeaturesRnn(nn.Module):
    def __init__(self, fbank_features, mfcc_features, hidden_size, batch_size=1):
        super(LibrosaFeaturesRnn, self).__init__()

        self.fbank_features = fbank_features
        self.mfcc_features = mfcc_features
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.fbank_lstm = nn.LSTM(fbank_features, hidden_size)
        self.mfcc_lstm  = nn.LSTM(mfcc_features, hidden_size)
        self.linear_both = nn.Linear(hidden_size * 2, 2)
        self.linear_mfcc = nn.Linear(hidden_size, 2)

    def init_hidden_state(self, device):
        return (torch.randn(1, self.batch_size, self.hidden_size).to(device),
                torch.randn(1, self.batch_size, self.hidden_size).to(device))

    def forward(self, x, device='cpu'):
        x_fbank = x[:, :, :self.fbank_features].permute(1, 0, 2)
        x_mfcc  = x[:, :, self.fbank_features:].permute(1, 0, 2)
        print(x_fbank.size())
        print(x_mfcc.size())

        out_fbank_lstm, _ = self.fbank_lstm(x_fbank, self.init_hidden_state(device))
        out_mfcc_lstm, _  = self.mfcc_lstm(x_mfcc,  self.init_hidden_state(device))

        out_mfcc = self.linear_mfcc(out_mfcc_lstm)
        out_both = self.linear_both(torch.cat([out_fbank_lstm, out_mfcc_lstm], dim=2))

        pred_mfcc = F.log_softmax(out_mfcc, dim=-1)
        pred_both = F.log_softmax(out_both, dim=-1)
        return pred_both, pred_mfcc


