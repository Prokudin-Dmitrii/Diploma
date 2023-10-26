import numpy as np
import pandas as pd
import seaborn as sns

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

class PositionalEmbedding(nn.Module):
  def __init__(self, d_model, max_seq_len=1000):

    # Позиционный энкодинг согласно AIAYN
    super(PositionalEmbedding, self).__init__()

    emb = torch.zeros(max_seq_len, d_model).float()
    emb.require_grad = False

    # d_model - размерность выходов эмбеддинга (кол-во каналов), должно быть чётным

    pos = torch.arange(0, max_seq_len).float()[:, None]
    div = (torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model)).exp()

    emb[:, 0::2] = torch.sin(pos * div)
    emb[:, 1::2] = torch.cos(pos * div)

    # Для работы с батчами
    emb = emb[None, :, :]
    self.emb = emb
  def forward(self, x):

    # Размерность входа: [Batch, Seq_Len, input_dim]

    # Размерность выхода: [1, X.shape[1], d_model]
    return self.emb[:, : x.shape[1]].float()

class ValueEmbedding(nn.Module):
  def __init__(self, input_dim, d_model, kernel_size=3):
    super(ValueEmbedding, self).__init__()
    self.kernel_size = kernel_size
    self.conv = nn.Conv1d(in_channels=input_dim, out_channels=d_model, kernel_size=kernel_size, padding=1, dtype=torch.float)

    # Правильная инициализация весов для ReLU
    for m in self.modules():
      if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
  
  def forward(self, x):
    # Размерность входа: [Batch, Seq_Len, input_dim]

    # Хотим [Batch, input_dim, Seq_Len]
    x = x.permute(0, 2, 1).float()

    # Размер выхода [Batch, Seq_Len, input_dim]
    result = self.conv(x).permute(0, 2, 1).float()
    return result.to(x.device, dtype=torch.float)


class TemporalEmbedding(nn.Module):
  def __init__(self, d_model):
    super(TemporalEmbedding, self).__init__()

    # Кодирование временных шагов

    #
    minute_size = 6
    hour_size = 24
    weekday_size = 7
    day_size = 32
    month_size = 13

    Embed = nn.Embedding
    self.minute_embed = Embed(minute_size, d_model)
    self.hour_embed = Embed(hour_size, d_model)
    self.weekday_embed = Embed(weekday_size, d_model)
    self.day_embed = Embed(day_size, d_model)
    self.month_embed = Embed(month_size, d_model)

  def forward(self, time):
    time = time.long()
    #print(time.shape)

    minutes = self.minute_embed(time[:, :, 0])
    hours = self.hour_embed(time[:, :, 1])
    days = self.day_embed(time[:, :, 2])
    weekdays = self.weekday_embed(time[:, :, 3])
    months = self.month_embed(time[:, :, 4])

    #return (minutes + hours + days + weekdays + months).float()
    return (hours + days + weekdays + months).float()

class DataEmbedding(nn.Module):
  def __init__(self, input_dim, d_model, kernel_size=3, max_seq_len=1000):
    super(DataEmbedding, self).__init__()
    self.vemb = ValueEmbedding(input_dim, d_model, kernel_size)
    self.pemb = PositionalEmbedding(d_model, max_seq_len)
    self.temp = TemporalEmbedding(d_model)
  
  def forward(self, x, x_time_stamps, use_time_stamps=False):
    #print(x.shape, x_time_stamps.shape)
    if use_time_stamps:
      x = self.vemb(x).to(x.device) + self.pemb(x).to(x.device) + self.temp(x_time_stamps)
    else:
      x = self.vemb(x).to(x.device) + self.pemb(x).to(x.device)
    return x

class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


class Inception_Block_V2(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels // 2):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=[1, 2 * i + 3], padding=[0, i + 1]))
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=[2 * i + 3, 1], padding=[i + 1, 0]))
        kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels + 1):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res
    


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, seq_len, pred_len, top_k, d_model, d_ff):
        super(TimesBlock, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.k = top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(d_model, d_ff),
            nn.GELU(),
            Inception_Block_V1(d_ff, d_model)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, seq_len, label_len, pred_len, input_dim, c_out, top_k=5, d_model=256, d_ff=32, emb_kern=3, max_seq_len=1000, e_layers=2):
        super(Model, self).__init__()

        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.model = nn.ModuleList([TimesBlock(seq_len, pred_len, top_k, d_model, d_ff)
                                    for _ in range(e_layers)])
        self.enc_embedding = DataEmbedding(input_dim=input_dim, d_model=d_model, kernel_size=emb_kern, max_seq_len=max_seq_len)
        self.layer = e_layers
        self.layer_norm = nn.LayerNorm(d_model)
        self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forecast(self, x_enc, x_mark_enc=None):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1)  # align temporal dimension
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # project back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        """
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        """
        return dec_out


    def forward(self, x_enc):
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
