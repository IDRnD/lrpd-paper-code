import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def get_file_list_from_folder(root, glob_expr="**/*.wav"):
    root = Path(root).resolve()
    return list(
        tqdm(root.glob(glob_expr), desc=f"retrieving files from : {root} by expr : '{glob_expr}'"))


def get_shallow_file_dict(root: str, glob_expr="**/*.wav") -> Dict[str, List[Path]]:
    d = {}
    root = Path(root).resolve()
    for subfolder in tqdm(root.glob("*/"), desc=f"building class dict for : {root}"):
        if subfolder.is_dir():
            d[subfolder.name] = list(subfolder.glob(glob_expr))
    num_samples_per_folder = list(map(len, d.values()))
    print(
        f"{root} :\n   min num samples : {np.min(num_samples_per_folder)}\n   max num samples : {np.max(num_samples_per_folder)}\n   mean num samples : {np.mean(num_samples_per_folder)}"
    )
    return d


def flatten_shallow_file_dict(data_dict: Dict[object, List[List[Path]]]):
    flttened = []
    for domain, domain_files in data_dict.items():
        flttened.extend(zip(domain_files, [domain] * len(domain_files)))
    return flttened


def keys_to_ints(d):
    sorted_keys = list(sorted(d.keys()))
    new_dict = {}
    for ind, key in enumerate(sorted_keys):
        new_dict[ind] = d[key]
    return new_dict


def merge_dicts(dicts, rebuild_keys=True):
    merged = {}
    current_ind = 0
    for d in dicts:
        for k in sorted(d.keys()):
            if rebuild_keys:
                merged[current_ind] = d[k]
            else:
                if k in merged:
                    warnings.warn(f"dicts have intersecting key : {k}")
                merged[k] = d[k]
            current_ind += 1
    return merged


def test_train_split_shallow_file_dict(data_dict: Dict[int, List[List[Path]]],
                                       test_size: float = 0.1,
                                       max_test_train_ratio: float = 0.3):
    test_data_dict = defaultdict(list)
    train_data_dict = defaultdict(list)

    for cls_ind, cls_domains in data_dict.items():
        for domain_files in cls_domains:
            if isinstance(test_size, float):
                test_size_ratio = test_size
            elif isinstance(test_size, int):
                test_size_ratio = test_size / len(domain_files)
            if max_test_train_ratio is not None:
                test_size_ratio = test_size_ratio if test_size_ratio < max_test_train_ratio else max_test_train_ratio
            train_files, test_files = train_test_split(domain_files, test_size=test_size_ratio)
            train_data_dict[cls_ind].append(train_files)
            test_data_dict[cls_ind].append(test_files)

    return train_data_dict, test_data_dict
