import torch
import torch.nn as nn
import models.erfnet as erfnet


class BranchedERFNet(nn.Module):
    def __init__(self, num_classes, encoder=None):
        super().__init__()

        print('Creating branched erfnet with {} classes'.format(num_classes))

        if (encoder is None):
            self.encoder = erfnet.Encoder(sum(num_classes))
        else:
            self.encoder = encoder

        self.decoders = nn.ModuleList()
        for n in num_classes:
            self.decoders.append(erfnet.Decoder(n))
###############在译码器后通过损失函数调整权重和偏移量的值##################################
    def init_output(self, n_sigma=1):
        with torch.no_grad():
            output_conv = self.decoders[0].output_conv
            print('initialize last layer with size: ',
                  output_conv.weight.size())

            output_conv.weight[:, 0:2, :, :].fill_(0)
            output_conv.bias[0:2].fill_(0)

            output_conv.weight[:, 2:2+n_sigma, :, :].fill_(0)
            output_conv.bias[2:2+n_sigma].fill_(1)

    def forward(self, input, only_encode=False):
        if only_encode:
            return self.encoder.forward(input, predict=True)
        else:
            output = self.encoder(input)

        return torch.cat([decoder.forward(output) for decoder in self.decoders], 1)
