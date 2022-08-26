from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

import nets
from losses import SupConLoss


class ViRLModel(nn.Module):

    def __init__(self, hidden_dim, rho, lr):
        super().__init__()
        self.rho = rho
        self.hidden_dim = hidden_dim

        self.conv = nets.ConvNet(hidden_dim)
        self.deconv = nets.DeconvNet(hidden_dim)
        self.lstm_enc = nets.LSTMEncoder(hidden_dim)
        self.lstm_dec = nets.LSTMDecoder(hidden_dim)

        self.conv_opt = torch.optim.Adam(self.conv.parameters(), lr)
        self.deconv_opt = torch.optim.Adam(self.deconv.parameters(), lr)
        self.lstm_enc_opt = torch.optim.Adam(self.lstm_enc.parameters(), lr)
        self.lstm_dec_opt = torch.optim.Adam(self.lstm_dec.parameters(), lr)

        self.contrast_loss = SupConLoss()

    def encode(self, video):
        video = video.unsqueeze(0)  # 1 x T x c x h x w
        video = video.to(dtype=torch.float) / 255.  # 1 x T x c x h x w
        video = torch.transpose(video, dim0=0, dim1=1)  # T x 1 x c x h x w
        e_seq = self._encode(video)  # T x 1 x h
        h_seq, _ = self.lstm_enc(e_seq)  # T x 1 x h
        e_seq = e_seq.squeeze(dim=1)
        h_seq = h_seq.squeeze(dim=1)
        return e_seq, h_seq

    def _encode(self, video):
        e_seq = []
        for t in range(video.shape[0]):
            e = self.conv(video[t])
            e_seq.append(e)
        e_seq = torch.stack(e_seq)
        return e_seq

    def _decode(self, e_seq):
        T = e_seq.shape[0]
        video = []
        for t in range(T):
            o = self.deconv(e_seq[t])
            video.append(o)
        video = torch.stack(video)
        return video

    def evaluate(self, video_i, video_p, video_n):
        T = video_i.shape[1]
        n = video_i.shape[0]

        video_i /= 255.  # n x T x c x h x w
        video_p /= 255.  # n x T x c x h x w
        video_n /= 255.  # n x T x c x h x w

        video_i = torch.transpose(video_i, dim0=0, dim1=1)  # T x n x c x h x w
        video_p = torch.transpose(video_p, dim0=0, dim1=1)  # T x n x c x h x w
        video_n = torch.transpose(video_n, dim0=0, dim1=1)  # T x n x c x h x w

        e_i_seq = self._encode(video_i)
        e_p_seq = self._encode(video_p)
        e_n_seq = self._encode(video_n)

        h_i_seq, hidden_i = self.lstm_enc(e_i_seq)
        h_p_seq, hidden_p = self.lstm_enc(e_p_seq)
        h_n_seq, hidden_n = self.lstm_enc(e_n_seq)

        e0_i_seq = self.lstm_dec(h_i_seq, T)
        e0_p_seq = self.lstm_dec(h_p_seq, T)

        video0_i = self._decode(e0_i_seq)
        video0_p = self._decode(e0_p_seq)
        video1_i = self._decode(e_i_seq)
        video1_p = self._decode(e_p_seq)

        l_sns = self.loss_sns(h_i_seq[-1], h_p_seq[-1], h_n_seq[-1])
        l_sni = self.loss_sni(e_i_seq, e_p_seq, e_n_seq)
        l_raes = self.loss_vae(video_i, video0_i) + self.loss_vae(video_p, video0_p)
        l_vaei = self.loss_vae(video_i, video1_i) + self.loss_vae(video_p, video1_p)
        loss = 0.7 * l_sns
        loss += 0.1 * l_sni
        loss += 0.1 * l_raes
        loss += 0.1 * l_vaei

        metrics = {
            'loss': loss.item(),
            'l_sns': l_sns.item(),
            'l_sni': l_sni.item(),
            'l_raes': l_raes.item(),
            'l_vaei': l_vaei.item()
        }

        return metrics, loss

    def update(self, video_i, video_p, video_n):
        self.train()

        self.conv_opt.zero_grad()
        self.lstm_enc_opt.zero_grad()
        self.lstm_dec_opt.zero_grad()
        self.deconv_opt.zero_grad()

        metrics, loss = self.evaluate(video_i, video_p, video_n)

        loss.backward()

        self.deconv_opt.step()
        self.lstm_dec_opt.step()
        self.lstm_enc_opt.step()
        self.conv_opt.step()

        self.eval()
        return metrics

    def loss_sns(self, h_i, h_p, h_n):
        return F.mse_loss(h_i, h_p) + max(self.rho - F.mse_loss(h_i, h_n), 0.)

    def loss_sni(self, e_seq_i, e_seq_p, e_seq_n):
        T = e_seq_i.shape[0]
        l = 0.
        for t in range(T):
            l += F.mse_loss(e_seq_i[t], e_seq_p[t]) + max(self.rho - F.mse_loss(e_seq_i[t], e_seq_n[t]), 0.)
        l /= T
        return l

    def loss_vae(self, video1, video2):
        T = video1.shape[0]
        l = 0.
        for t in range(T):
            o1 = video1[t].flatten(start_dim=1)
            o2 = video2[t].flatten(start_dim=1)
            l += F.mse_loss(o1, o2)
        l /= T
        return l

    @staticmethod
    def load(file):
        snapshot = Path(file)
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        return payload['encoder']



