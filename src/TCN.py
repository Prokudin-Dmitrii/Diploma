import numpy as np
import pandas as pd
import seaborn as sns

import torch
from torch import nn
from torch.nn import functional as f
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils import weight_norm

"""
  При padding`e добавляются элементы слева и справа последовательности, поэтому лишние элементы необходимо обрезать
"""

class Chomp1d(nn.Module):
  def __init__(self, chomp_size):
      super(Chomp1d, self).__init__()
      self.chomp_size = chomp_size

  def forward(self, x):
      return x[:, :, :-self.chomp_size].contiguous()

"""
  Основной блок TCN, представляет из себя residual блок и состоит из 2 dilated causal convolution слоёв,
  2 активаций (ReLU), weight_norm в качестве нормализации и dropout
  : Параметр n_inputs: количество входных каналов
  : Параметр n_outputs: int, количество выходных каналов
  : Параметр kernel_size: int, размер ядра свёртки
  : Параметр Stride: int, шаг для stride, зафиксирован 1 в TCN
  : Параметр Dilation: int, коэффициент расширения. It is related to the number of layers where this Residual Block (or, hidden layer) is located. 
                                                                 For example, if this Residual Block is on the first layer, dilation = 2 ** 0 = 1;
                                                                             If this Residual Block is on the 2nd layer, dilation = 2 ** 1 = 2;
                                                                             If this Residual Block is on the 3rd layer, dilation = 2 ** 2 = 4;
                                                                             If this Residual Block is on the 4th layer, dilation = 2 ** 3 = 8 ......
  : Параметр padding: int, filling coefficient. Связан с размером ядра и коэффициентов расширения. 
  : Параметр Dropout: Float, частота dropout.

"""

class TemporalBlock(nn.Module):
  def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
    super(TemporalBlock, self).__init__()
    self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                          stride=stride, padding=padding, dilation=dilation))
    self.chomp1 = Chomp1d(padding)
    self.relu1 = nn.ReLU()
    self.dropout1 = nn.Dropout(dropout)

    self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                          stride=stride, padding=padding, dilation=dilation))
    self.chomp2 = Chomp1d(padding)
    self.relu2 = nn.ReLU()
    self.dropout2 = nn.Dropout(dropout)

    self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                              self.conv2, self.chomp2, self.relu2, self.dropout2)
        
#"""
#  1 × 1 свёртка. Используется, когда количество входных каналов отличается от количества выходных в Temporal блоке.
#  Обычно они различны, пока значение в num_channels не равно значению num_inputs. Например, [5, 5, 5, 5] и num_inputs = 5
#"""
    self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

#"""
#  Нелинейность при помощи функции активации, действует на весь Temporal блок, можно убрать/менять/игнорировать.
#"""
    self.relu = nn.ReLU()
    self.init_weights()

  def init_weights(self):
    self.conv1.weight.data.normal_(0, 0.01)
    self.conv2.weight.data.normal_(0, 0.01)
    if self.downsample is not None:
      self.downsample.weight.data.normal_(0, 0.01)

  def forward(self, x):
    out = self.net(x)
    res = x if self.downsample is None else self.downsample(x)
    return self.relu(out + res)

"""
 num_levels - глубина сети, определяет количество Temporal блоков, определяется длиной num_channels
 num_channels - массив значений количества выходных каналов, например: [5, 5, 5] - три блока, выход с каждого 5 каналов
 num_inputs - количество входных каналов
 kernel_size - размер ядра
 dropout - dropout
"""

class TemporalConvNet(nn.Module):
  def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
    super(TemporalConvNet, self).__init__()
    layers = []
    num_levels = len(num_channels)
    for i in range(num_levels):
      dilation_size = 2 ** i
      in_channels = num_inputs if i == 0 else num_channels[i-1]
      out_channels = num_channels[i]
      layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                    padding=(kernel_size-1) * dilation_size, dropout=dropout)]

    self.network = nn.Sequential(*layers)

  def forward(self, x):
    return self.network(x)


import torch.nn.functional as F

"""
  input_size === num_inputs - количество входных каналов
  output_size - размер выхода с линейного слоя (классификатора), т.е., для MNIST, например, 10
  num_channels, kernel_size, dropout - см. выше

  Полная модель TCN состоит из n Temporal блоков и одного линейного слоя 
"""


class TCN(nn.Module):
  def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
    super(TCN, self).__init__()
    self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
    self.linear = nn.Linear(num_channels[-1], output_size)

  def forward(self, inputs):
    #    """Inputs have to have dimension (N, C_in, L_in)"""
    y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
    o = self.linear(y1[:, :, -1])
    return F.log_softmax(o, dim=1)

"""
#parser = argparse.ArgumentParser(description='Sequence Modeling - (Permuted) Sequential MNIST')
#parser.add_argument('--batch_size', type=int, default=64, metavar='N',
#                    help='batch size (default: 64)')
#parser.add_argument('--cuda', action='store_false',
#                    help='use CUDA (default: True)')
#parser.add_argument('--dropout', type=float, default=0.05,
#                    help='dropout applied to layers (default: 0.05)')
#parser.add_argument('--clip', type=float, default=-1,
#                    help='gradient clip, -1 means no clip (default: -1)')
#parser.add_argument('--epochs', type=int, default=20,
#                    help='upper epoch limit (default: 20)')
#parser.add_argument('--ksize', type=int, default=7,
#                    help='kernel size (default: 7)')
#parser.add_argument('--levels', type=int, default=8,
#                    help='# of levels (default: 8)')
#parser.add_argument('--log-interval', type=int, default=100, metavar='N',
#                    help='report interval (default: 100')
#parser.add_argument('--lr', type=float, default=2e-3,
#                    help='initial learning rate (default: 2e-3)')
#parser.add_argument('--optim', type=str, default='Adam',
#                   help='optimizer to use (default: Adam)')
#parser.add_argument('--nhid', type=int, default=25,
#                    help='number of hidden units per layer (default: 25)')
#parser.add_argument('--seed', type=int, default=1111,
#                    help='random seed (default: 1111)')
#parser.add_argument('--permute', action='store_true',
#                    help='use permuted MNIST (default: false)')
#args = parser.parse_args()
"""


class TCN_TSF(nn.Module):
  def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
    super(TCN_TSF, self).__init__()
    self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
    self.linear_0 = nn.Linear(num_channels[-1], output_size)
    self.linear_1 = nn.Linear(num_channels[-1], output_size)

  def forward(self, inputs):
    #    """Inputs have to have dimension (N, C_in, L_in)"""
    y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
    out_0 = self.linear_0(y1[:, :, -1])
    out_1 = self.linear_1(y1[:, :, -1])
    out_0 = out_0[:, None, :]
    out_1 = out_1[:, None, :]
    return torch.cat((out_0, out_1), 1)
  

from torch.nn import MultiheadAttention

class TCN_Encoder(nn.Module):
  def __init__(self, input_channels=1, num_channels=[25]*8, kernel_size=7, dropout=0.05):
    super(TCN_Encoder, self).__init__()
    self.encoder = TemporalConvNet(num_inputs = input_channels, num_channels=num_channels, kernel_size=kernel_size, dropout=dropout)
  
  def forward(self, encoder_input):
    # Размер входа для Encoder: [N, C, L]
    # Размер выхода Encoder: [N, C, L]
    return self.encoder(encoder_input)

class TCN_Decoder(nn.Module):
  def __init__(self, input_channels=1, num_channels=[25]*8, kernel_size=7, dropout=0.05, num_heads=5, layer_norm=True, device='cpu'):
    super(TCN_Decoder, self).__init__()
    self.decoder = TemporalConvNet(num_inputs = input_channels, num_channels=num_channels, kernel_size=kernel_size, dropout=dropout)
    self.self_attention = MultiheadAttention(embed_dim=num_channels[-1], num_heads=num_heads, batch_first=True)
    self.attention = MultiheadAttention(embed_dim=num_channels[-1], num_heads=num_heads, batch_first=True)
    self.layer_norm = layer_norm
    if self.layer_norm:
      self.norm = nn.BatchNorm1d(num_features=num_channels[-1])
    else:
      self.norm = nn.BatchNorm1d(num_features=num_channels[-1])

    self.device = device

  def create_temp_mask(self, size):
    mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


  def forward(self, encoder_output, target, average_attn_weights=True):
    # Размер входа для Decoder: [N, C, L]
    decoder_output = self.decoder(target)
    # Размер выхода Decoder: [N, C, L]
    decoder_output = decoder_output.permute(0, 2, 1)
    encoder_output = encoder_output.permute(0, 2, 1)
    # Размер входа для MHA: [N, L, C]
    mask = self.create_temp_mask(decoder_output.shape[1]).to(self.device)

    self_attention_output, self_attn = self.self_attention(query=decoder_output, value=decoder_output, key=decoder_output, attn_mask=mask, average_attn_weights=average_attn_weights)

    decoder_output = decoder_output + self_attention_output

    attention_output, cross_attn = self.attention(query=decoder_output, value=encoder_output, key=encoder_output, average_attn_weights=average_attn_weights)

    # Размер входа для Norm: [N, C, L]
    output = decoder_output + attention_output
    output = output.permute(0, 2, 1)

    result = self.norm(output)

    return result, (self_attn, cross_attn)

class TCN_EncDecAttn(nn.Module):
  def __init__(self, classes=2, input_channels=1, num_channels=[25]*8, kernel_size=7, dropout=0.05, device='cpu', num_heads=14):
    super(TCN_EncDecAttn, self).__init__()
    self.device = device
    self.encoder = TCN_Encoder(input_channels = input_channels, num_channels=num_channels, kernel_size=kernel_size, dropout=dropout)
    self.decoder = TCN_Decoder(input_channels = input_channels, num_channels=num_channels, kernel_size=kernel_size, dropout=dropout, device=self.device, num_heads=num_heads)
    self.linear = nn.Linear(num_channels[-1], classes)

  def forward(self, encoder_input, target, average_attn_weights=True):
    # Вход [N, L, C], но нужен [N, C, L]
    encoder_input = encoder_input.permute(0, 2, 1)
    target = target.permute(0, 2, 1)
    encoder_output = self.encoder(encoder_input)
    result, (self_attn, cross_attn) = self.decoder(encoder_output, target, average_attn_weights=average_attn_weights)
    result = result.permute(0, 2, 1)

    result = self.linear(result)
    #Выход [N, L, C]

    return result, (self_attn, cross_attn)

class AutoRegressiveTCN(nn.Module):
  def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
    super(AutoRegressiveTCN, self).__init__()
    self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
    self.linear = nn.Linear(num_channels[-1], output_size)

  def forward(self, inputs):
    # Вход должен быть размером (Batch, Channels, Seq_Length)
    # Но вход, скорее всего, будет (Batch, Seq_Length, Channels)
    inputs = inputs.permute(0, 2, 1)

    output = self.tcn(inputs)  # inputs: (Batch, Channels, Seq_Length)
    # output: (Batch, num_channels[-1], Seq_Length)
    output = output.permute(0, 2, 1)
    # Для линейного слоя вход должен быть (Batch, Seq_Length, num_channels[-1])
    result = self.linear(output)
    # result: (Batch, Seq_Length, output_size)
    return result


class TCN_TSF_Perm(nn.Module):
  def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
    super(TCN_TSF_Perm, self).__init__()
    self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
    self.linear = nn.Linear(num_channels[-1], output_size)

  def forward(self, inputs):
    #    """Inputs have to have dimension (N, C_in, L_in)"""
    y = self.tcn(inputs)  # input should have dimension (N, C, L)
    out = self.linear(y[:, :, -1])
    return out


class TCN_EncDecAttn_ETT(nn.Module):
  def __init__(self, input_channels=1, num_channels=[25]*8, kernel_size=7, dropout=0.05, device='cpu', num_heads=14):
    super(TCN_EncDecAttn_ETT, self).__init__()
    self.device = device
    self.encoder = TCN_Encoder(input_channels = input_channels, num_channels=num_channels, kernel_size=kernel_size, dropout=dropout)
    self.decoder = TCN_Decoder(input_channels = input_channels, num_channels=num_channels, kernel_size=kernel_size, dropout=dropout, device=self.device, num_heads=num_heads)
    self.linear = nn.Linear(num_channels[-1], input_channels * 2)

  def forward(self, encoder_input, target, average_attn_weights=True):
    # Вход [N, L, input_channels], но нужен [N, input_channels, L]
    encoder_input = encoder_input.permute(0, 2, 1)
    target = target.permute(0, 2, 1)
    encoder_output = self.encoder(encoder_input)
    result, (self_attn, cross_attn) = self.decoder(encoder_output, target, average_attn_weights=average_attn_weights)
    result = result.permute(0, 2, 1)

    result = self.linear(result)
    #Выход [N, L, input_channels * 2]

    return result, (self_attn, cross_attn)

class AutoRegressiveTCN_ETT(nn.Module):
  def __init__(self, input_size, num_channels, kernel_size, dropout):
    super(AutoRegressiveTCN_ETT, self).__init__()
    self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
    self.linear = nn.Linear(num_channels[-1], input_size * 2)

  def forward(self, inputs):
    # Вход должен быть размером (Batch, Channels, Seq_Length)
    # Но вход, скорее всего, будет (Batch, Seq_Length, Channels)
    inputs = inputs.permute(0, 2, 1)

    output = self.tcn(inputs)  # inputs: (Batch, Channels, Seq_Length)
    # output: (Batch, num_channels[-1], Seq_Length)
    output = output.permute(0, 2, 1)
    # Для линейного слоя вход должен быть (Batch, Seq_Length, num_channels[-1])
    result = self.linear(output)
    # result: (Batch, Seq_Length, input_size * 2)
    return result