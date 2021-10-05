from typing import Dict, NewType, Tuple, Union

import torch
import torch.nn as nn

# Load all model builders
from .poolings.stats import STAT_POOLINGS
from .architectures.rawnet import RawNet

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


def build_sequential_fcnet(
    input_features_chan: int,
    layers=[(256, 0.5, "ReLU")],
):
    sequential = []
    for ind, (num_units, dropout_rate, activ) in enumerate(layers):
        sequential.append(nn.Linear(input_features_chan, num_units, bias=True))
        sequential.append(nn.BatchNorm1d(num_units))
        input_features_chan = num_units
        if activ is not None:
            sequential.append(getattr(nn, activ)())
        if dropout_rate > 0:
            sequential.append(nn.Dropout(p=dropout_rate))
    return nn.Sequential(*sequential)


class MultiTaskClassificationHead(nn.Module):
    def __init__(
            self,
            head_setups: Dict[str, Union[int, Tuple[int, float], Dict[str, object]]],
            input_features_chan: int,
            head_hidden_layers=[
                # (num_units, dropout, activation)
                (256, 0.5, "ReLU"),
            ],
            return_embeddings=False):
        super(MultiTaskClassificationHead, self).__init__()
        sequential = []
        self.return_embeddings = return_embeddings
        shared_hidden_layers = head_hidden_layers
        self.fc_net = build_sequential_fcnet(input_features_chan, shared_hidden_layers)

        def generate_head(head_setup, input_features_chan):
            dropout_rate = 0.0
            head_layers = []
            if isinstance(head_setup, int):
                num_classes = head_setup
            elif isinstance(head_setup, tuple) or isinstance(head_setup, list):
                num_classes, dropout_rate = head_setup
            elif isinstance(head_setup, dict):
                num_classes = head_setup["num_classes"]
                dropout_rate = head_setup.get("dropout_rate", 0.0)
                if "hidden_layers" in head_setup:
                    head_layers.append(
                        build_sequential_fcnet(input_features_chan, head_setup["hidden_layers"]))
                    input_features_chan = head_setup["hidden_layers"][-1][0]
            if dropout_rate > 0.0:
                head_layers.append(nn.Dropout(p=dropout_rate))
            head_layers.append(nn.Linear(input_features_chan, num_classes, bias=True))
            return nn.Sequential(*head_layers)

        input_features_chan = shared_hidden_layers[-1][0]
        self.heads = nn.ModuleDict(
            modules={
                head_name: generate_head(head_setup, input_features_chan)
                for head_name, head_setup in head_setups.items()
            })

    def forward(self, x):
        x = self.fc_net(x)
        if self.return_embeddings:
            return dict([(head_name, self.heads[head_name](x))
                         for head_name in self.heads.keys()]), x
        else:
            return dict([(head_name, self.heads[head_name](x)) for head_name in self.heads.keys()])


class MultiInMultiOut(nn.Module):
    def __init__(
        self,
        branches_setup=[
            {
                "input_path": [0, -1],  # for example
                "output_name": "denoising_mask_logits",
                "module": object
            },
            {
                "input_path": [0, -1],  # for example
                "output_name": "vad_logits",
                "module": object
            },
        ]):
        super(MultiInMultiOut, self).__init__()
        module_dict = {}
        input_map = {}
        for branch_setup in branches_setup:
            branch_name = branch_setup["output_name"]
            branch_input_path = branch_setup["input_path"]
            input_map[branch_name] = branch_input_path
            module_dict[branch_name] = resolve_model_or_conf(branch_setup["module"])
        self.module_dict = nn.ModuleDict(modules=module_dict)
        print(self.modules)
        self.input_map = input_map

    def forward(self, inputs):
        outputs = {}
        for branch_name in self.input_map.keys():
            x = inputs
            if self.input_map[branch_name] is None:
                x = x
            else:
                for path_part in self.input_map[branch_name]:
                    x = x[path_part]
            outputs[branch_name] = self.module_dict[branch_name](x)
        return outputs


class StatsPooling2D(nn.Module):
    def __init__(self, mode="var"):
        super(StatsPooling2D, self).__init__()
        self.convert_mode_on = False
        self.mode = mode

    def forward(self, x):
        s = x.size()
        # x = x.view(s[0],s[1]*s[2],s[3])
        # x = x.reshape(s[0],s[1]*s[2],s[3])
        # x = torch.reshape(x,(int(s[0]),int(s[1]*s[2]),int(s[3])))
        if self.convert_mode_on:
            print(f"RESHAPE -> SPLIT+CONCAT")
            x = torch.cat(torch.split(x, 1, dim=1), dim=2)[:, 0, :, :]
        else:
            x = torch.reshape(x, (int(s[0]), int(s[1] * s[2]), int(s[3])))
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


class Permute(nn.Module):
    def __init__(self, permutation):
        super(StatsPooling1D, self).__init__()
        self.permutation = permutation

    def forward(self, x):
        return x.permute(self.permutation)


class Transpose(nn.Module):
    def __init__(self, perm):
        super(Transpose, self).__init__()
        self.perm = perm

    def forward(self, x):
        return x.transpose(self.perm)


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


class NormalizeAudio(nn.Module):
    def __init__(self):
        super(NormalizeAudio, self).__init__()

    def forward(self, x):
        val_range = (x.max(dim=-1, keepdims=True).values - x.min(dim=-1, keepdims=True).values +
                     1e-8) / 2
        return (x - x.mean(dim=-1, keepdims=True)) / val_range


class AudioClassificationModel(nn.Module):
    def __init__(self,
                 features: ModuleOrConfig = None,
                 backbone: ModuleOrConfig = None,
                 pooling: ModuleOrConfig = None,
                 cls_head: ModuleOrConfig = None,
                 spec_augs: ModuleOrConfig = None):
        super(AudioClassificationModel, self).__init__()
        self.features = resolve_model_or_conf(features)
        self.backbone = resolve_model_or_conf(backbone)
        self.pooling = resolve_model_or_conf(pooling)
        self.cls_head = resolve_model_or_conf(cls_head)
        self.spec_augs = resolve_model_or_conf(spec_augs)

    def forward(self, x):
        if self.features is not None:
            x = self.features(x)
        if self.spec_augs is not None:
            if self.training:
                with torch.no_grad():
                    x = self.spec_augs(x)
        if self.backbone is not None:
            x = self.backbone(x)
        if self.pooling is not None:
            x = self.pooling(x)
        if self.cls_head is not None:
            x = self.cls_head(x)
        return x


class SequentialModel(nn.Module):
    def __init__(self, submodules=[]):
        super(SequentialModel, self).__init__()
        self.submodules = nn.Sequential(*[resolve_model_or_conf(sm) for sm in submodules])

    def forward(self, x):
        return self.submodules(x)
