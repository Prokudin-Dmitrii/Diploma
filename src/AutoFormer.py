import numpy as np
import pandas as pd
import seaborn as sns

import torch
from torch import nn
from torch.nn import functional as f
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

class AutoCorrelation(nn.Module):
  # Модуль автокорреляции, замена attention слоям, имеет те же параметры, т.е. они взаимозаменяемы
  # Имеется две фазы:
  # 1) period-based dependencies discovery
  # 2) time delay aggregation

  def __init__(self, factor=3):
    super(AutoCorrelation, self).__init__()
    self.factor = factor

  def time_delay_aggregation_train(self, values, corr):
    pass

  def time_delay_aggregation_infer(self, values, corr):
    pass
  
  def time_delay_aggregation(self, values, corr):

    # Есть ускоренные варианты, но я их писать не буду (ПОКА ЧТО)

    # Вход размера [B, H, C, L]
    batch, head, channels, length = values.shape

    # Инициализация индексов
    # [1, 1, 1, Length]
    # Для каждого батча, каждой головы и каждого канала повторяем набор индексов
    # [B, H, C, Length]
    index_init = torch.arange(length).repeat(batch, head, channels, 1).to(values.device)

    # Поиск top-k 
    # k = Целое снизу от (c * log(L)), где L - длина последовательности, c - гиперпараметр слоя (в статье перебирали от 1 до 3)
    top_k = int((self.factor * np.log(length)))
    # Возвращает туплю (values, indices)
    # weights - Значения авто-корреляции, полученные из FFT и IFFT: [tok_k]
    # delay - значения временных периодов соотвествующих корреляций, числа от 0 до L-1 (соотв. периоды от 1 до L): [top_k]
    weights, delay = torch.topk(corr, top_k, dim=-1)

    # Обновляем corr - делаем веса/коэффициенты для аггрегации из их "уверенности"
    # Чем больше значение авто-корреляции, тем мы более уверены, что есть такая периодичность
    temp_corr = torch.softmax(weights, dim=-1)

    # Aggregation
    # Values - ряд, который начинаем разворачивать на период времени Тау из top_k:
    # Тау шагов из начала переносятся в конец
    # Затем полученный ряд умножается на его вес из temp_corr
    temp_values = values.repeat(1, 1, 1, 2)

    # Заготовка для финальной суммы
    delays_agg = torch.zeros_like(values).float()

    for i in range(top_k):
      # Получили новые индексы для сдвинутого ряда (как раз перенос начала в конец на нужный период)
      # delay[..., i]
      temp_delay = index_init + delay[..., i].unsqueeze(-1)

      # Держим индексы в нужных рамках
      temp_delay = temp_delay % 8

      # Получаем сдвинутые ряды из изначальных с помощью новых индексов
      pattern = torch.gather(temp_values, dim=-1, index=temp_delay)

      # Аггрегируем полученный ряд, умноженный на соответсвующий вес
      delays_agg = delays_agg + pattern * (temp_corr[..., i].unsqueeze(-1))
      """

      temp_delay = index_init + delay[..., i].unsqueeze(-1)
      pattern = torch.gather(temp_values, dim=-1, index=temp_delay)
      delays_agg = delays_agg + pattern * (temp_corr[..., i].unsqueeze(-1))
      """
    # Возвращаем аггрегированный результат
    # Размер как у values: [B, H, C, L]
    return delays_agg

  def forward(self, queries, keys, values):
    batch, length, head, e_ch = queries.shape
    _, seq, _, d_ch = values.shape
    if length > seq:
      zeros = torch.zeros_like(queries[:, :(length - seq), :]).float()
      values = torch.cat((values, zeros), dim=1)
      keys = torch.cat((keys, zeros), dim=1)
    else:
      values = values[:, : length, :, :]
      keys = keys[:, : length, :, :]
    
    # Находим периодичные зависимости с помощью FFT и IFFT
    query_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
    keys_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)

    result = query_fft * torch.conj(keys_fft)

    corr = torch.fft.irfft(result, n=length, dim=-1)

    # Аггрегация
    return (self.time_delay_aggregation(values.permute(0, 2, 3, 1).contiguous(), corr)).contiguous(), corr.permute(0, 3, 1, 2)

class AutoCorrelationModule(nn.Module):
  def __init__(self, d_model, n_heads, d_keys=None, d_values=None, factor=3):
    super(AutoCorrelationModule, self).__init__()

    if d_keys is None:
      d_keys = d_model // n_heads
    
    if d_values is None:
      d_values = d_model // n_heads
    
    # Автокорреляционный модуль
    self.corr = AutoCorrelation(factor=factor)

    # Проекции на нужные размерности каналов
    self.query_proj = nn.Linear(d_model, d_keys * n_heads, dtype=torch.float)
    self.values_proj = nn.Linear(d_model, d_values * n_heads, dtype=torch.float)
    self.keys_proj = nn.Linear(d_model, d_keys * n_heads, dtype=torch.float)
    self.out_proj = nn.Linear(d_values * n_heads, d_model, dtype=torch.float)

    self.n_heads = n_heads

  def forward(self, queries, keys, values):
    # queries размер [Batch, Length, Channels]

    # keys, values размер [Batch, Length либо seq, Channels] ???
    batch, length, channels = queries.shape
    _, seq, _ = keys.shape
    heads = self.n_heads

    queries = self.query_proj(queries).reshape(batch, length, heads, -1)
    keys = self.keys_proj(keys).view(batch, seq, heads, -1)
    values = self.values_proj(values).view(batch, seq, heads, -1)

    result, corr = self.corr(queries, keys, values)

    result = result.view(batch, length, -1)

    return self.out_proj(result), corr

class moving_avg(nn.Module):
  # AvgPool1d Не умеет в паддинги какими-то конкретными числами с каждого конца, поэтому можно реализовать
  pass

class Series_Decomposition(nn.Module):
  # Разложени ряда в сезонную и трендовую части:
  def __init__(self, kernel_size=25):
    super(Series_Decomposition, self).__init__()
    self.avg = nn.AvgPool1d(kernel_size=kernel_size, padding=((kernel_size - 1)//2), stride=1)
  
  def forward(self, x):
    # Вход размера [B, L, C]
    # Для AvgPool1d нужен [B, C, L], его выход тоже [B, C, L]
    trend = self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)
    season = x - trend
    return season, trend

class Special_Layernorm(nn.Module):
  """
  Special designed layernorm for the seasonal part
  """
  def __init__(self, d_model):
    super(Special_Layernorm, self).__init__()
    self.layernorm = nn.LayerNorm(d_model)

  def forward(self, x):
    x_hat = self.layernorm(x)
    bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
    return x_hat - bias

class Encoder(nn.Module):
  def __init__(self, d_model, correlation, d_ff=None, kernel_size=25, dropout=0.1, activation='relu'):
    super(Encoder, self).__init__()
    if d_ff is None:
      d_ff = 4 * d_model
    # Часть автокорреляции
    self.corr = correlation

    # Две 1д свёртки в качестве Feed Forward слоя в соотв. с AIAYN
    self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, dtype=torch.float)
    self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, dtype=torch.float)

    # Два декомпозишина
    self.sd1 = Series_Decomposition(kernel_size)
    self.sd2 = Series_Decomposition(kernel_size)

    # Дропаут для красоты
    self.drop = nn.Dropout(dropout)
    self.activation = nn.ReLU() if activation == 'relu' else nn.GELU()
  
  def forward(self, x):

    # Размерность входа [B, L, C]
    result, corr = self.corr(x, x, x)
    x = x + self.drop(result)

    x, _ = self.sd1(x)
    # Свёрткам нужно [B, C, L]
    result = x.permute(0, 2, 1)
    result = self.drop(self.activation(self.conv1(result)))
    result = self.drop(self.conv2(result)).permute(0, 2, 1)

    # Снова [B, L, C]
    x = x + result
    
    x,_ = self.sd2(x)
    
    return x, corr

class EncoderModule(nn.Module):
  def __init__(self, d_model, correlation, d_ff=None, kernel_size=21, dropout=0.1, activation='relu', num_layers=2):
    super(EncoderModule, self).__init__()
    m_list = []
    for i in range(num_layers):
      m_list.append(Encoder(d_model=d_model, correlation=correlation, d_ff=d_ff, kernel_size=kernel_size,dropout=dropout,activation=activation))
    self.enc = nn.ModuleList(m_list)

    self.norm = Special_Layernorm(d_model)

  def forward(self, encoder_input):
    corrs = []
    for i in self.enc:
      encoder_input, corr = i(encoder_input)
      corrs.append(corr)
    
    encoder_input = self.norm(encoder_input)
    
    return encoder_input, corrs

class Decoder(nn.Module):
  def __init__(self, d_model, correlation, cross_correlation, c_out=1, d_ff=None, kernel_size=21, dropout=0.1, activation='relu'):
    super(Decoder, self).__init__()
    
    if d_ff is None:
      d_ff = 4 * d_model
    # Часть автокорреляции
    self.corr = correlation
    self.cross_corr = cross_correlation

    # Две 1д свёртки в качестве Feed Forward слоя в соотв. с AIAYN
    self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, dtype=torch.float)
    self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, dtype=torch.float)

    # Два декомпозишина
    self.sd1 = Series_Decomposition(kernel_size)
    self.sd2 = Series_Decomposition(kernel_size)
    self.sd3 = Series_Decomposition(kernel_size)

    # Дропаут для красоты
    self.drop = nn.Dropout(dropout)
    self.activation = nn.ReLU() if activation == 'relu' else nn.GELU()

    self.proj = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, padding=1, bias=False, dtype=torch.float)
  
  def forward(self, decoder_input, encoder_output):
    result, _ = self.corr(decoder_input, decoder_input, decoder_input)
    decoder_input = decoder_input + self.drop(result)

    decoder_input, x_det1 = self.sd1(decoder_input)
    result, _ = self.cross_corr(decoder_input, encoder_output, encoder_output)
    decoder_input = decoder_input + self.drop(result)
    decoder_input, x_det2 = self.sd2(decoder_input)
    
    result = decoder_input
    result = self.drop(self.activation(self.conv1(result.transpose(-1, 1))))
    result = self.drop(self.conv2(result).transpose(-1, 1))

    decoder_input, x_det3 = self.sd3(decoder_input + result)

    trend = x_det1 + x_det2 + x_det3
    trend = self.proj(trend.permute(0, 2, 1)).transpose(1, 2)

    return decoder_input, trend

class DecoderModule(nn.Module):
  def __init__(self, d_model, correlation, cross_correlation, c_out=1, d_ff=None, kernel_size=21, dropout=0.1, activation='relu', num_layers=1):
    super(DecoderModule, self).__init__()
    m_list = []
    for i in range(num_layers):
      m_list.append(Decoder(d_model=d_model, correlation=correlation, cross_correlation=cross_correlation, c_out=c_out, d_ff=d_ff,\
                              kernel_size=kernel_size,dropout=dropout, activation=activation))
    self.dec = nn.ModuleList(m_list)

    self.proj = nn.Linear(d_model, c_out, bias=True)
    self.norm = Special_Layernorm(d_model)

  def forward(self, decoder_input, encoder_output, trend_init=None):
    for i in self.dec:
      decoder_input, trend = i(decoder_input, encoder_output)
      trend_init += trend
    
    decoder_input = self.norm(decoder_input)
    
    decoder_input = self.proj(decoder_input)
    
    return decoder_input, trend_init
  
class DecoderModuleTSF(nn.Module):
  def __init__(self, d_model, correlation, cross_correlation, c_out=1, d_ff=None, kernel_size=21, dropout=0.1, activation='relu', num_layers=1):
    super(DecoderModuleTSF, self).__init__()
    m_list = []
    for i in range(num_layers):
      m_list.append(Decoder(d_model=d_model, correlation=correlation, cross_correlation=cross_correlation, c_out=c_out, d_ff=d_ff,\
                              kernel_size=kernel_size,dropout=dropout, activation=activation))
    self.dec = nn.ModuleList(m_list)

    self.proj = nn.Linear(c_out, d_model, bias=True, dtype=torch.double)
    self.norm = Special_Layernorm(d_model)

  def forward(self, decoder_input, encoder_output, trend_init=None):
    for i in self.dec:
      decoder_input, trend = i(decoder_input, encoder_output)
      trend_init += trend
    
    decoder_input = self.norm(decoder_input)
    
    trend_init = self.proj(trend_init)
    
    return decoder_input, trend_init

class AutoFormer(nn.Module):
  def __init__(self, seq_len, label_len, pred_len, input_dim, d_model=24, emb_kern=3, max_seq_len=2000, kernel_size=25, \
               d_ff=None, dropout=0.05, activation='relu', enc_layers=2, dec_layers=1, n_heads=8, d_keys=None, d_values=None, factor=3):
    super(AutoFormer, self).__init__()

    self.seq_len = seq_len
    self.label_len = label_len
    self.pred_len = pred_len

    self.embedding = DataEmbedding(input_dim=input_dim, d_model=d_model, kernel_size=emb_kern, max_seq_len=max_seq_len)
    self.sd = Series_Decomposition(kernel_size)

    self.encoder = EncoderModule(d_model=d_model, correlation=AutoCorrelationModule(d_model, n_heads, d_keys, d_values, factor),\
                                 d_ff=d_ff, kernel_size=kernel_size, dropout=dropout, activation=activation, num_layers=enc_layers)
    self.decoder = DecoderModule(d_model=d_model, correlation=AutoCorrelationModule(d_model, n_heads, d_keys, d_values, factor),\
                                 cross_correlation=AutoCorrelationModule(d_model, n_heads, d_keys, d_values, factor),\
                                 c_out=input_dim, d_ff=d_ff, kernel_size=kernel_size, dropout=dropout, activation=activation, num_layers=dec_layers)
    
    #self.season_proj = nn.Linear(d_model, input_dim, dtype=torch.float)

  def forward(self, x, x_time_stamps=None, use_time_stamps=False):
        mean = torch.mean(x, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x.shape[0], self.pred_len, x.shape[2]], device=x.device)
        seasonal_init, trend_init = self.sd(x)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)

        if use_time_stamps:
          enc_out = self.embedding(x, x_time_stamps[:, : self.seq_len, :], use_time_stamps)
        else:
          enc_out = self.embedding(x, None, use_time_stamps)
        enc_out, attns = self.encoder(enc_out)
        # dec
        if use_time_stamps:
          dec_out = self.embedding(seasonal_init, x_time_stamps[:, self.seq_len - self.label_len: self.seq_len + self.pred_len, :], use_time_stamps)
        else:
          dec_out = self.embedding(seasonal_init, None, use_time_stamps)

        seasonal_part, trend_part = self.decoder(dec_out, enc_out, trend_init=trend_init)

        dec_out = trend_part + seasonal_part
        
        return dec_out[:, -self.pred_len:, :], attns # [B, L, D_INP]

class AutoFormerTSF(nn.Module):
  def __init__(self, seq_len, label_len, pred_len, input_dim, d_model=24, emb_kern=3, max_seq_len=2000, kernel_size=25, \
               d_ff=None, dropout=0.05, activation='relu', enc_layers=2, dec_layers=1, n_heads=8, d_keys=None, d_values=None, factor=3):
    super(AutoFormerTSF, self).__init__()

    self.seq_len = seq_len
    self.label_len = label_len
    self.pred_len = pred_len

    self.embedding = DataEmbedding(input_dim=input_dim, d_model=d_model, kernel_size=emb_kern, max_seq_len=max_seq_len)
    self.sd = Series_Decomposition(kernel_size)

    self.encoder = EncoderModule(d_model=d_model, correlation=AutoCorrelationModule(d_model, n_heads, d_keys, d_values, factor),\
                                 d_ff=d_ff, kernel_size=kernel_size, dropout=dropout, activation=activation, num_layers=enc_layers)
    self.decoder = DecoderModuleTSF(d_model=d_model, correlation=AutoCorrelationModule(d_model, n_heads, d_keys, d_values, factor),\
                                 cross_correlation=AutoCorrelationModule(d_model, n_heads, d_keys, d_values, factor),\
                                 c_out=input_dim, d_ff=d_ff, kernel_size=kernel_size, dropout=dropout, activation=activation, num_layers=dec_layers)
    
    #self.season_proj = nn.Linear(d_model, input_dim, dtype=torch.float)

  def forward(self, x, x_time_stamps, use_time_stamps=False):
        mean = torch.mean(x, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x.shape[0], self.pred_len, x.shape[2]], device=x.device)
        seasonal_init, trend_init = self.sd(x)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)

        if use_time_stamps:
          enc_out = self.embedding(x, x_time_stamps[:, : self.seq_len, :], use_time_stamps)
        else:
          enc_out = self.embedding(x, None, use_time_stamps)
        enc_out, attns = self.encoder(enc_out)
        # dec
        if use_time_stamps:
          dec_out = self.embedding(seasonal_init, x_time_stamps[:, self.seq_len - self.label_len: self.seq_len + self.pred_len, :], use_time_stamps)
        else:
          dec_out = self.embedding(seasonal_init, None, use_time_stamps)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, trend_init=trend_init)

        dec_out = trend_part + seasonal_part
        
        return dec_out[:, -self.pred_len:, :], attns # [B, L, d_model]

#model = AutoFormer(783, 783, 28, 21)
#y, _= model(torch.ones(64, 784, 21), torch.ones(64, 784 * 2, 21))
#print(y.shape)

class AutoFormerETT(nn.Module):
  def __init__(self, pred_len, input_dim, seq_len=None, label_len=None, d_model=24, emb_kern=3, max_seq_len=2000, kernel_size=21, \
               d_ff=None, dropout=0.1, activation='relu', enc_layers=2, dec_layers=1, n_heads=8, d_keys=None, d_values=None, factor=3):
    super(AutoFormerETT, self).__init__()

    if label_len is None:
      label_len = pred_len
    if seq_len is None:
      seq_len = pred_len
    self.autoformer = AutoFormerTSF(seq_len, label_len, pred_len, input_dim, d_model, emb_kern, max_seq_len, kernel_size,\
               d_ff, dropout, activation, enc_layers, dec_layers, n_heads, d_keys, d_values, factor)
    
    self.linear = nn.Linear(d_model, input_dim * 2, dtype=torch.double)

  def forward(self, x, x_time_stamps):
    # x: [B, SEQ_LEN, INPUT_DIM]
    # x_time_stamps: [B, SEQ_LEN + PRED_LEN, 5]
    autoformer_result, attns = self.autoformer(x, x_time_stamps) # [B, PRED_LEN, INPUT_DIM]
    
    results = self.linear(autoformer_result) # [B, PRED_LEN, INPUT_DIM * 2]
    return results, attns
  
class AutoFormerSWaT(nn.Module):
  def __init__(self, pred_len, input_dim, classes, seq_len=None, label_len=None, d_model=24, emb_kern=3, max_seq_len=2000, kernel_size=21, \
               d_ff=None, dropout=0.1, activation='relu', enc_layers=2, dec_layers=1, n_heads=8, d_keys=None, d_values=None, factor=3):
    super(AutoFormerSWaT, self).__init__()

    if label_len is None:
      label_len = pred_len
    if seq_len is None:
      seq_len = pred_len
    self.autoformer = AutoFormerTSF(seq_len, label_len, pred_len, input_dim, d_model, emb_kern, max_seq_len, kernel_size,\
               d_ff, dropout, activation, enc_layers, dec_layers, n_heads, d_keys, d_values, factor)
    
    self.linear = nn.Linear(d_model, classes, dtype=torch.double)

  def forward(self, x):
    # x: [B, SEQ_LEN, INPUT_DIM]
    # x_time_stamps: [B, SEQ_LEN + PRED_LEN, 5]
    autoformer_result, attns = self.autoformer(x, None) # [B, PRED_LEN, INPUT_DIM]
    
    results = self.linear(autoformer_result) # [B, PRED_LEN, CLASSES]
    return results, attns