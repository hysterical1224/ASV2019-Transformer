import torch.nn as nn
from otrans.encoder import TransformerEncoder


class Transformer(nn.Module):
    def __init__(self, params):
        super(Transformer, self).__init__()

        self.params = params
        # input_size=512, nfft=40, kernel_size=1024, device='cuda'
        self.encoder = TransformerEncoder(input_size=params['feat_dim'],
                                          nfft = params['nfft'],
                                          kernel_size=params['kernel_size'],
                                          d_model=params['d_model'],
                                          attention_heads=params['n_heads'],
                                          linear_units=params['enc_ffn_units'],
                                          num_blocks=params['num_enc_blocks'],
                                          pos_dropout_rate=params['pos_dropout_rate'],
                                          slf_attn_dropout_rate=params['slf_attn_dropout_rate'],
                                          ffn_dropout_rate=params['ffn_dropout_rate'],
                                          residual_dropout_rate=params['residual_dropout_rate'],
                                          input_layer=params['enc_input_layer'],
                                          normalize_before=params['normalize_before'],
                                          concat_after=params['concat_after'],
                                          activation=params['activation'],
                                          device=params['device']
                                          )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(params['d_model']),
            nn.Linear(params['d_model'], params['num_classes'])
        )
        self.pool = 'cls'
        self.to_latent = nn.Identity()
        self.sig = nn.Sigmoid()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs):
        x = self.encoder(inputs)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)

        x = self.mlp_head(x)

        output = self.logsoftmax(x)

        return output