from unittest import TestCase

import torch

from Wavenet import WaveNet
from torch import nn
from einops import repeat


class TestWaveNet(TestCase):

    def setUp(self) -> None:
        super().setUp()
        input_channels = 1
        out_channels = 1
        sampleSize = 2000
        batch_size = 8
        self.wavenet = WaveNet(input_channels, out_channels, kernel_size=2, stack_size=2, layer_size=5)
        self.input = nn.Parameter(torch.randn((input_channels, sampleSize)))
        self.input = repeat(self.input, 'c t->b c t', b=batch_size)

    def test_runWavenet(self):
        output = self.wavenet(self.input)
        pass

    def test_calculateReceptiveFieldAndOutputSize(self):
        print(self.wavenet.calculateReceptiveField())
        print(self.wavenet.calculateOutputSize(self.input))
