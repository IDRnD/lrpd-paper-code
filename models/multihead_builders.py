import sys
from pathlib import Path
from typing import Dict, NewType, Union

import numpy as np
import torch
import torch.nn as nn


root_path = Path(__file__).resolve().parents[1]
if str(root_path) not in sys.path:
    print(f"Adding pipeline tf2 root in sys.path: {root_path}")
    sys.path.append(str(root_path))

# from models.architectures.clova_resnet import ClovaResNetSE
# from models.architectures.magneto import MagNetOResNet
# from models.architectures.rawnet import RawNet
# from models.features.spectrograms import MelSpecAug, SpectrogramAug
# from models.features.spectrograms_tf import SpectralFeaturesTF
from models.poolings.stats import STAT_POOLINGS


class ClassificationHead(nn.Module):
    def __init__(self,
                 num_classes: int,
                 input_features_chan: int,
                 head_hidden_layers=[
                     (256, 0.5, "ReLU"),
                 ]):
        super(ClassificationHead, self).__init__()
        input_channels = input_features_chan
        sequential = []
        for ind, (num_units, dropout_rate, activ) in enumerate(head_hidden_layers):
            sequential.append(nn.Linear(input_features_chan, num_units, bias=True))
            sequential.append(nn.BatchNorm1d(num_units))
            input_features_chan = num_units
            if activ is not None:
                sequential.append(getattr(nn, activ)())
            if dropout_rate > 0:
                sequential.append(nn.Dropout(p=dropout_rate))
        if num_classes is not None:
            sequential.append(nn.Linear(head_hidden_layers[-1][0], num_classes, bias=True))
        self.fc_net = nn.Sequential(*sequential)

    def forward(self, x):
        return self.fc_net(x)


class StatsPooling2D(nn.Module):
    def __init__(self, mode="var"):
        super(StatsPooling2D, self).__init__()
        self.mode = mode

    def forward(self, x):
        s = x.size()
        # x = x.view(s[0],s[1]*s[2],s[3])
        x = x.reshape(x.size()[0], -1, x.size()[-1])
        x = STAT_POOLINGS[self.mode](x, dim=2)
        return x


class StatsPooling1D(nn.Module):
    def __init__(self, mode="var"):
        super(StatsPooling1D, self).__init__()
        self.mode = mode

    def forward(self, x):
        s = x.size()
        x = STAT_POOLINGS[self.mode](x, dim=2)
        return x


def get_params_count(model: nn.Module):
    trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params_count = sum(p.numel() for p in model.parameters())
    return trainable_params_count, all_params_count


ModuleOrConfig = NewType("Strides", Union[nn.Module, Dict])


def resolve_model_or_conf(mod_or_conf: ModuleOrConfig):
    print(mod_or_conf)
    if mod_or_conf is None:
        return mod_or_conf
    if isinstance(mod_or_conf, dict):
        module = eval(mod_or_conf["type"])(**mod_or_conf["params"])
        trainable = mod_or_conf.get("trainable", True)
        if trainable is not None:
            for param in module.parameters():
                param.requires_grad = trainable
        print(
            f"{mod_or_conf['type']} trainable {trainable}, params counts : {get_params_count(module)}"
        )
        return module
    elif isinstance(mod_or_conf, nn.Module):
        return mod_or_conf
    else:
        raise NotImplemented()


class MultiheadModel(nn.Module):
    def __init__(
        self,
        features: ModuleOrConfig = {
            "type":
            "MelSpecAug",
            "params":
            dict(
                extend_spec_channels=1,
                use_specaug=False,
                f_min=20.,
                f_max=7500.,
                n_fft=512,
                win_length=400,  # 0.025 sec * 16000
                hop_length=160,  # 0.010 sec * 16000
                n_mels=80,
                log_mels=True,
                sample_rate=16000,
            ),
        },
        backbone: ModuleOrConfig = {
            "type":
            "MagNetOResNet",
            "params":
            dict(
                init_conv_params=dict(
                    in_channels=1,
                    out_channels=32,
                    stride=1,
                    kernel_size=3,
                    padding=1,
                ),
                block="BasicBlock",
                block_setup=[
                    # filters, num_blocks, strides
                    (32, 32, 3, 1),
                    (32, 64, 4, 2),
                    (64, 128, 6, 2),
                    (128, 256, 3, 2)
                ],
                norm_layer=None)
        },
        pooling: ModuleOrConfig = {
            "type": "StatsPooling2D",
            "params": dict()
        },
        cls_head_phone: ModuleOrConfig = {
            "type":
            "ClassificationHead",
            "params":
            dict(input_features_chan=256 * 10 * 2,
                 num_classes=15,
                 head_hidden_layers=[
                     (256, 0.0, "ReLU"),
                 ])
        },
        cls_head_speaker: ModuleOrConfig = {
            "type":
            "ClassificationHead",
            "params":
            dict(input_features_chan=256 * 10 * 2,
                 num_classes=8,
                 head_hidden_layers=[
                     (256, 0.0, "ReLU"),
                 ])
        }):
        super(MultiheadModel, self).__init__()
        self.features = resolve_model_or_conf(features)
        self.backbone = resolve_model_or_conf(backbone)
        self.pooling = resolve_model_or_conf(pooling)
        self.cls_head_phone = resolve_model_or_conf(cls_head_phone)
        self.cls_head_speaker = resolve_model_or_conf(cls_head_speaker)

    def forward(self, x):
        if self.features is not None:
            x = self.features(x)

        if self.backbone is not None:
            x = self.backbone(x)

        if self.pooling is not None:
            x = self.pooling(x)

        x_phone = self.cls_head_phone(x)
        x_speaker = self.cls_head_speaker(x)

        return x_phone, x_speaker


if __name__ == "__main__":
    model = MultiheadModel()
    model.eval()

    input = torch.from_numpy(np.random.rand(3, 1, 48000).astype(np.float32))
    output = model(input)

    print("output phones shape", output[0].shape)
    print("output speaker shape", output[1].shape)
