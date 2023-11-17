from typing import Optional, Tuple, Union, NamedTuple

import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from torchvision.models.convnext import CNBlock, LayerNorm2d
from torchvision.models.swin_transformer import SwinTransformerBlock
from dataclasses import dataclass
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import ModelOutput


class PatchEmbedding(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, output_channels // 2, kernel_size=kernel_size, stride=stride,
                               padding=padding)
        self.norm1 = nn.BatchNorm2d(output_channels // 2)
        self.conv2 = nn.Conv2d(output_channels // 2, output_channels, kernel_size=kernel_size, stride=stride,
                               padding=padding)
        self.norm2 = nn.BatchNorm2d(output_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = nn.functional.relu(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = nn.functional.relu(x)

        return x


class Encoder(nn.Module):
    def __init__(self, Ci: list[int]):
        super().__init__()
        self.Ci = Ci

        self.patch_embedding = PatchEmbedding(3, self.Ci[0])  # H/4 * W/4 * C0

        self.stage_1 = nn.Sequential(
            *[CNBlock(self.Ci[0], 1e-6, 0) for _ in range(1)]
        )

        self.downsampling_1 = nn.Sequential(
            # LayerNorm2d(C0, eps=1e-6),
            nn.Conv2d(self.Ci[0], self.Ci[1], kernel_size=(2, 1), stride=(2, 1))
        )  # H/8 * W/4 * C1

        self.stage_2 = nn.Sequential(
            *[CNBlock(self.Ci[1], 1e-6, 0) for _ in range(1)]
        )

        self.downsampling_2 = nn.Sequential(
            # LayerNorm2d(C1, eps=1e-6),
            nn.Conv2d(self.Ci[1], self.Ci[2], kernel_size=(2, 1), stride=(2, 1))
        )  # H/16 * W/4 * C2

        self.stage_3 = nn.Sequential(
            *[CNBlock(self.Ci[2], 1e-6, 0) for _ in range(1)]
        )

        self.downsampling_3 = nn.Sequential(
            # LayerNorm2d(C2, eps=1e-6),
            nn.Conv2d(self.Ci[2], self.Ci[3], kernel_size=(2, 2), stride=(2, 2))
        )  # H/32 * W/8 * C3

        self.stage4 = nn.Sequential(
            *[SwinTransformerBlock(self.Ci[3], 4, [5, 40], [0, 0]) for _ in range(1)]
        )

        self.downsampling_4 = nn.Sequential(
            # LayerNorm2d(C3, eps=1e-6),
            nn.Conv2d(self.Ci[3], self.Ci[4], kernel_size=(1, 2), stride=(1, 2))
        )  # H/32 * W/16 * C4

        self.stage5 = nn.Sequential(
            *[SwinTransformerBlock(self.Ci[4], 4, [5, 40], [0, 0]) for _ in range(1)]
        )

        self.downsampling_5 = nn.Sequential(
            # LayerNorm2d(C4, eps=1e-6),
            nn.Conv2d(self.Ci[4], self.Ci[5], kernel_size=(1, 2), stride=(1, 2))
        )  # H/32 * W/32 * C5

        self.stage6 = nn.Sequential(
            *[SwinTransformerBlock(self.Ci[5], 4, [5, 40], [0, 0]) for _ in range(1)]
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.stage_1(x)
        x = self.downsampling_1(x)
        x = self.stage_2(x)
        x = self.downsampling_2(x)
        x = self.stage_3(x)
        x = self.downsampling_3(x)

        x = x.permute(0, 2, 3, 1)
        x = self.stage4(x)
        x = x.permute(0, 3, 1, 2)
        x = self.downsampling_4(x)
        x = x.permute(0, 2, 3, 1)
        x = self.stage5(x)
        x = x.permute(0, 3, 1, 2)
        x = self.downsampling_5(x)
        x = x.permute(0, 2, 3, 1)
        x = self.stage6(x)
        x = x.permute(0, 3, 1, 2)

        x = x.permute(0, 2, 3, 1)
        x = torch.reshape(x, (1, -1, self.Ci[5]))

        return x


class Decoder(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        num_features = num_features
        decoder_layer = TransformerDecoderLayer(d_model=num_features, nhead=4)
        self.decoder = TransformerDecoder(decoder_layer, 1)

    def forward(self, encoder_output, decoder_sequence):
        x = self.decoder(decoder_sequence, encoder_output)
        return x


from transformers import VisionEncoderDecoderConfig, AutoConfig, VisionEncoderDecoderModel, AutoModel, RobertaTokenizer, \
    TFBertTokenizer


class DocParserConfig(PretrainedConfig):
    model_type = "docparser"

    def __init__(
            self,
            ci: list[int],
            **kwargs
    ):
        super().__init__(**kwargs)
        self.ci = ci



@dataclass
class DocParserOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    # pooler_output: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor]] = None



class DocParserModel(nn.Module):
    def __init__(self, config: DocParserConfig):
        super().__init__()
        self.config = config

        self.encoder_embedding = PatchEmbedding(3, self.config.ci[0])
        self.encoder = Encoder(self.config.ci)

        # self.decoder_embedding =
        # self.decoder = Decoder(self.config.ci[-1])

    def _from_config(config: DocParserConfig):
        return DocParserModel(config)

    def forward(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, DocParserOutput]:
        print("ok on commence")
        print(pixel_values.shape)
        encoder_output = self.encoder(pixel_values)
        # decoder_sequence = torch.randn(1, 256, 1024)
        # x = self.decoder(encoder_output, decoder_sequence)
        output = DocParserOutput(
            last_hidden_state=encoder_output,
        )
        print(output[0].shape)
        return output

    def get_output_embeddings(self):
        return None

