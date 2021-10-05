import json
from pprint import pprint

import torch
import torch.nn as nn

from .model_builders import AudioClassificationModel


def set_convert_mode_on(model, flag=True):
    for name, child in model.named_children():
        if hasattr(child, "convert_mode_on"):
            print(f"{child} convert_mode_on")
            child.convert_mode_on = flag
        set_convert_mode_on(child, flag=flag)


def load_weights_from_pl_pipeline(net,
                                  weights_path: str,
                                  key_replace_dict={"nnet.": ""},
                                  remove_unessacary: bool = True,
                                  strict: bool = True,
                                  map_loc: str = None):
    # Change keys of state_dict so they can be used inside
    # bare torch model from builder not from pytorch-ligthning module

    # 0. Changes keys in state_dict according to key_replace_dict
    if map_loc is not None:
        state_dict = torch.load(weights_path, map_location=torch.device(map_loc))['state_dict']
    else:
        state_dict = torch.load(weights_path)['state_dict']

    for k in list(state_dict.keys()):
        old_k = str(k)
        for replace_what, replace_by in key_replace_dict.items():
            k = k.replace(replace_what, replace_by)
        if old_k != k:
            state_dict[k] = state_dict[old_k]
        del state_dict[old_k]

    pretrained_dict = state_dict
    if remove_unessacary:
        model_dict = net.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. load the new state dict
    missing_keys, unexpected_keys = net.load_state_dict(pretrained_dict, strict=strict)
    print(f"unexpected_keys : {unexpected_keys}")
    print(f"missing_keys : {missing_keys}")


def set_batchnorms_momentum(model: nn.Module, momentum: float = 0.99):
    layers = torch.flatten(model)
    for l in layers:
        if isinstance(l, nn.BatchNorm1d) or isinstance(l, nn.BatchNorm2d):
            l.momentum = momentum


def load_classification_model_from_experiment(experiment_dir, epoch):
    model_config = json.loads((experiment_dir / "model_config.json").read_text())
    model = AudioClassificationModel(**model_config)
    model.cls_head.return_embeddings = True
    model = model.eval()
    weights_path = list((experiment_dir / "checkpoints").glob(f"epoch={epoch}-step=*"))[0]
    print(weights_path)
    load_weights_from_pl_pipeline(model, str(weights_path), remove_unessacary=False, strict=False)
    return model
