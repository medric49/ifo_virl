import torch
from torch import nn


class ConvNet(nn.Module):
    def __init__(self, hidden_dim):
        super(ConvNet, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=5, stride=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, kernel_size=5, stride=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, hidden_dim * 4, kernel_size=1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dim * 4, hidden_dim, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, obs):
        e = self.network(obs)
        e = e.view(e.shape[0], e.shape[1])
        return e


class DeconvNet(nn.Module):
    def __init__(self, hidden_dim):
        super(DeconvNet, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 4, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dim * 4, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 3, kernel_size=5, stride=2, output_padding=1),
        )

    def forward(self, e):
        e = e.view(e.shape[0], e.shape[1], 1, 1)
        obs = self.network(e)
        return obs


class LSTMEncoder(nn.Module):
    def __init__(self, input_size):
        super(LSTMEncoder, self).__init__()
        self.num_layers = 2
        self.encoder = nn.LSTM(input_size=input_size, hidden_size=input_size, num_layers=self.num_layers)
        self.fc = nn.Linear(input_size, input_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, e_seq):
        T = e_seq.shape[0]
        h_seq, hidden = self.encoder(e_seq)
        h_seq = torch.stack([self.sigmoid(self.fc(h_seq[i])) for i in range(T)])
        return h_seq, hidden


class LSTMDecoder(nn.Module):
    def __init__(self, input_size):
        super(LSTMDecoder, self).__init__()
        self.num_layers = 2
        self.decoder = nn.LSTM(input_size=input_size, hidden_size=input_size, num_layers=self.num_layers)
        self.fc = nn.Linear(input_size, input_size)

    def forward(self, h_seq, T):
        h = h_seq[-1]
        h = self.fc(h)
        e_seq = []

        hidden = None
        for _ in range(T):
            h = h.unsqueeze(0)
            h, hidden = self.decoder(h, hidden)
            h = h.squeeze(0)
            e_seq.append(h)
        e_seq = torch.stack(e_seq)
        return e_seq
