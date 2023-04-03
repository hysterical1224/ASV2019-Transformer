import torch
import torch.nn as nn
import torch.nn.functional as F
from otrans.module import *
from otrans.utils import get_enc_padding_mask
from otrans.layer import TransformerEncoderLayer
import numpy as np
from torch import Tensor
import otrans.impulse_responses as impulse_responses

class SincConv(nn.Module):
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, device, out_channels, kernel_size, in_channels=1, sample_rate=16000,
                 stride=1, padding=0, dilation=1, bias=False, groups=1):

        super(SincConv, self).__init__()

        if in_channels != 1:
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        self.device = device
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        # initialize filterbanks using Mel scale
        NFFT = 2048
        f = int(self.sample_rate / 2) * np.linspace(0, 1, int(NFFT / 2) + 1)
        fmel = self.to_mel(f)  # Hz to mel conversion
        fmelmax = np.max(fmel)
        fmelmin = np.min(fmel)
        filbandwidthsmel = np.linspace(fmelmin, fmelmax, self.out_channels + 1)
        filbandwidthsf = self.to_hz(filbandwidthsmel)  # Mel to Hz conversion
        self.mel = filbandwidthsf
        self.hsupp = torch.arange(-(self.kernel_size - 1) / 2, (self.kernel_size - 1) / 2 + 1)
        self.band_pass = torch.zeros(self.out_channels, self.kernel_size)
        # print("self.band_pass:",self.band_pass.shape)
        #[20, 1025]

    def forward(self, x):
        for i in range(len(self.mel) - 1):
            fmin = self.mel[i]
            fmax = self.mel[i + 1]
            hHigh = (2 * fmax / self.sample_rate) * np.sinc(2 * fmax * self.hsupp / self.sample_rate)
            hLow = (2 * fmin / self.sample_rate) * np.sinc(2 * fmin * self.hsupp / self.sample_rate)
            hideal = hHigh - hLow

            self.band_pass[i, :] = Tensor(np.hamming(self.kernel_size)) * Tensor(hideal)
        # print("self.band_pass:",self.band_pass.shape)
        # [20, 1025]

        band_pass_filter = self.band_pass.to(self.device)

        self.filters = (band_pass_filter).view(self.out_channels, 1, self.kernel_size)
        # print("self.filters:",self.filters.shape)
        # [20, 1, 1025]

        # [12, 1, 30000]
        # print("in:",x.shape)
        x = F.conv1d(x, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                        bias=None, groups=1)
        x = x.transpose(1,2)
        # print("sinc:",x.shape)
        return x

# class GaussianLowpass(nn.Module):
#     """Depthwise pooling (each input filter has its own pooling filter).
#
#     Pooling filters are parametrized as zero-mean Gaussians, with learnable
#     std. They can be initialized with tf.keras.initializers.Constant(0.4)
#     to approximate a Hanning window.
#     We rely on depthwise_conv2d as there is no depthwise_conv1d in Keras so far.
#     """
#
#     def __init__(
#         self,
#         kernel_size,
#         strides=1,
#         nfft=64,
#         padding=0,
#         use_bias=True,
#         kernel_initializer=nn.init.xavier_uniform_,
#         kernel_regularizer=None,
#         trainable=False):
#
#         super(GaussianLowpass, self).__init__()
#         self.kernel_size = kernel_size
#         self.strides = strides
#         self.nfft = nfft
#         self.padding = padding
#         self.use_bias = use_bias
#         self.kernel_initializer = kernel_initializer
#         self.kernel_regularizer = kernel_regularizer
#         self.trainable = trainable
#
#         initialized_kernel = self.kernel_initializer(torch.zeros(1, 1, self.nfft, 1).type(torch.float32))
#         self._kernel = nn.Parameter(initialized_kernel, requires_grad=self.trainable)
#
#         # Register an initialization tensor here for creating the gaussian lowpass impulse response to automatically
#         # handle cpu/gpu device selection.
#         self.register_buffer("gaussian_lowpass_init_t", torch.arange(0, self.kernel_size, dtype=torch.float32))
#
#     def forward(self, x):
#         kernel = impulse_responses.gaussian_lowpass(self._kernel, self.kernel_size, self.gaussian_lowpass_init_t)
#         kernel = kernel.squeeze(3)
#         kernel = kernel.permute(2, 0, 1)
#
#         outputs = F.conv1d(x, kernel, stride=self.strides, groups=self.nfft, padding=self.padding)
#         return outputs

class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super(Residual_block, self).__init__()
        self.first = first

        if not self.first:
            self.bn1 = nn.BatchNorm1d(num_features=nb_filts[0])

        self.lrelu = nn.LeakyReLU(negative_slope=0.3)

        self.conv1 = nn.Conv1d(in_channels=nb_filts[0],
                               out_channels=nb_filts[1],
                               kernel_size=3,
                               padding=1,
                               stride=1)

        self.bn2 = nn.BatchNorm1d(num_features=nb_filts[1])
        self.conv2 = nn.Conv1d(in_channels=nb_filts[1],
                               out_channels=nb_filts[1],
                               padding=1,
                               kernel_size=3,
                               stride=1)

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv1d(in_channels=nb_filts[0],
                                             out_channels=nb_filts[1],
                                             padding=0,
                                             kernel_size=1,
                                             stride=1)

        else:
            self.downsample = False
        self.mp = nn.MaxPool1d(3)

        self.output_num = [4, 2, 1]

    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.lrelu(out)
        else:
            out = x

        out = self.conv1(x)
        out = self.bn2(out)
        out = self.lrelu(out)
        out = self.conv2(out)

        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        out = spatial_pyramid_pool(x, 1, [int(x.size(-2)), int(x.size(-1))], self.output_num)
        # out = self.mp(out)
        return out

nb_filts = [40, [40, 40], [40, 128], [128, 128]]


class TransformerEncoder(nn.Module):

    def __init__(self, input_size, nfft, kernel_size, device='cuda', d_model=256, attention_heads=4, linear_units=2048, num_blocks=6, pos_dropout_rate=0.0,
                 slf_attn_dropout_rate=0.0, ffn_dropout_rate=0.0, residual_dropout_rate=0.1, input_layer="conv2d",
                 normalize_before=True, concat_after=False, activation='relu', type='transformer'):
        super(TransformerEncoder, self).__init__()



        self.device = device
        # kernel_size = 1024
        # out_channels = 20
        self.sinc_conv = SincConv(device=self.device,
                                  out_channels=nfft,
                                  kernel_size=kernel_size,
                                  in_channels=1)


        self.first_bn = nn.BatchNorm1d(num_features=nb_filts[0])
        self.selu = nn.SELU(inplace=True)
        self.block0 = nn.Sequential(Residual_block(nb_filts[1], first=True))
        self.block1 = nn.Sequential(Residual_block(nb_filts[1]))
        self.block2 = nn.Sequential(Residual_block(nb_filts[2]))
        nb_filts[2][0] = nb_filts[2][1]
        self.block3 = nn.Sequential(Residual_block(nb_filts[2]))
        self.block4 = nn.Sequential(Residual_block(nb_filts[2]))
        self.block5 = nn.Sequential(Residual_block(nb_filts[2]))
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.fc_attention0 = self._make_attention_fc(in_features=nb_filts[1][-1],
                                                     l_out_features=nb_filts[1][-1])
        self.fc_attention1 = self._make_attention_fc(in_features=nb_filts[1][-1],
                                                     l_out_features=nb_filts[1][-1])
        self.fc_attention2 = self._make_attention_fc(in_features=nb_filts[2][-1],
                                                     l_out_features=nb_filts[2][-1])
        self.fc_attention3 = self._make_attention_fc(in_features=nb_filts[2][-1],
                                                     l_out_features=nb_filts[2][-1])
        self.fc_attention4 = self._make_attention_fc(in_features=nb_filts[2][-1],
                                                     l_out_features=nb_filts[2][-1])
        self.fc_attention5 = self._make_attention_fc(in_features=nb_filts[2][-1],
                                                     l_out_features=nb_filts[2][-1])

        self.bn_before_gru = nn.BatchNorm1d(num_features=nb_filts[2][-1])


        # self.gauss = GaussianLowpass(kernel_size = kernel_size,nfft=nfft)

        self.normalize_before = normalize_before

        if input_layer == "linear":
            self.embed = LinearWithPosEmbedding(input_size, d_model, pos_dropout_rate)
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling(input_size, d_model, pos_dropout_rate)
        elif input_layer == 'conv2dv2':
            self.embed = Conv2dSubsamplingV2(input_size, d_model, pos_dropout_rate)

        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(attention_heads, d_model, linear_units, slf_attn_dropout_rate, ffn_dropout_rate,
                                    residual_dropout_rate=residual_dropout_rate, normalize_before=normalize_before,
                                    concat_after=concat_after, activation=activation) for _ in range(num_blocks)
        ])

        if self.normalize_before:
            self.after_norm = LayerNorm(d_model)

    def forward(self, inputs):
        nb_samp = inputs.shape[0]
        len_seq = inputs.shape[1]
        x=inputs.view(nb_samp,1,len_seq)
        x = self.sinc_conv(x)
        x = torch.abs(x)
        # inputs = F.max_pool1d(torch.abs(inputs), 3)
        # [1, 1024, 12715]
        # print("sinc_oup:", x.shape)
        # inputs = self.gauss(inputs)
        # print("Gauss_inputs:",inputs.shape)

        enc_mask = get_enc_padding_mask(x)
        # print("enc_mask.shape:",enc_mask.shape)
        # [12, 1, 20]
        enc_output, enc_mask = self.embed(x, enc_mask)
        # print("enc_output:",enc_output.shape)

        enc_output.masked_fill_(~enc_mask.transpose(1, 2), 0.0)

        for _, block in enumerate(self.blocks):
            enc_output, enc_mask = block(enc_output, enc_mask)
            enc_output.masked_fill_(~enc_mask.transpose(1, 2), 0.0)

        if self.normalize_before:
            enc_output = self.after_norm(enc_output)

        return enc_output

import torchaudio

if __name__ == '__main__':
    # x = torch.randn(6, 64600).to(torch.float32).cuda()
    # # sinc = SincConv(device='cuda',out_channels=20, kernel_size=1024, in_channels=1)
    # # output = sinc(x)
    # # print("output:",output.shape)
    # # x = torch.randn(1, 28976, 20).cuda()
    # model = TransformerEncoder(input_size=512, nfft=40, kernel_size=1024, device='cuda').to('cuda')
    # transformer_outputs = model(x)
    # print(transformer_outputs.shape)
    path = r"I:\2021\data\PA\ASVspoof2019_PA_train\flac\PA_T_0000001.flac"
    sig, sr = torchaudio.load_wav(path)
    compute_fbank = torchaudio.compliance.kaldi.fbank(torch.tensor(sig), num_mel_bins=64)
    compute_fbank = compute_fbank.unsqueeze(0).to(torch.float32).cuda()
    print(compute_fbank.shape)
    # model = TransformerEncoder(input_size=512, nfft=1024, kernel_size=1024, device='cuda').to('cuda')
    # transformer_outputs = model(compute_fbank, input_length=30000)
    # print(transformer_outputs.shape)

