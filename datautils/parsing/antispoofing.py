import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def parse_data_pickle(data_config: str, ext: str = "wav"):
    with open(data_config, "rb") as f:
        return pickle.load(f)


def parse_data_config(data_config: Dict[str, int], ext: str = "wav", **kwargs):
    data_dict = defaultdict(list)
    for subdomain_root in sorted(data_config.keys()):
        cls_ind = data_config[subdomain_root]
        subdomain_files = list(Path(subdomain_root).glob(f"**/*.{ext}"))
        print(f"{str(subdomain_root)} : {len(subdomain_files)}")
        data_dict[cls_ind].append(subdomain_files)
    return data_dict


def parse_dir(path, label, ext: str = "wav"):
    path = Path(path)
    files_dict = {label: list(path.glob(f"**/*.{ext}"))}
    return files_dict


def parse_asv17(asv_spoof_root: str, part: str = "dev", return_as="dict"):
    asv_spoof_root = Path(asv_spoof_root).resolve()
    wavs = [(wavp, wavp.name)
            for wavp in (asv_spoof_root / f"ASVspoof2017_V2_{part}").glob("*.wav")]
    protocol = pd.read_csv(
        str(asv_spoof_root / "protocol_V2" /
            f"ASVspoof2017_V2_{part}.{'trn' if part == 'train' else 'trl'}.txt"),
        sep=" ",
        names=["file", "label", "spk_id", "phrase_id", "env_id", "play_id", "rec_id"])
    if return_as is not None:
        wavs_dict = {fn: fp for fp, fn in wavs}
        files = []
        for _, row in protocol.iterrows():
            fn = row['file']
            fp = wavs_dict[fn]
            label = int(row["label"] == "spoof")
            files.append((fp, label))
        if return_as == "list":
            return files
        elif return_as == "dict":
            files_dict = defaultdict(list)
            for fp, label in files:
                files_dict[label].append(fp)
            return files_dict
    else:
        return wavs, protocol


def parse_data_config_v2(data_config: Dict[str, int],
                         ext: str = "wav",
                         depth=1,
                         exclude_substrings=[],
                         min_num_files: int = 2000):
    data_dict = defaultdict(list)
    for subdomain_root in sorted(data_config.keys()):
        cls_ind = data_config[subdomain_root]
        subdomain_root = Path(subdomain_root)
        for subsubdomain in subdomain_root.glob("*/" * depth):
            if not subsubdomain.is_dir():
                continue
            skip = False
            for es in exclude_substrings:
                if es in str(subsubdomain):
                    skip = True
                    print(
                        f"Skipping {subsubdomain} cause it consists exceluded substring : <{es}>")
            if skip:
                continue
            subsubdomain_files = list(subsubdomain.glob(f"**/*.{ext}"))
            if len(subsubdomain_files) > min_num_files:
                print(f"{str(subsubdomain)} : {len(subsubdomain_files)}")
                data_dict[cls_ind].append(subsubdomain_files)
            else:
                print(f"{str(subsubdomain)} : {len(subsubdomain_files)} < {min_num_files}")

        subdomain_files = list(subsubdomain.glob(f"*.{ext}"))
        if len(subdomain_files) > min_num_files:
            print(f"{str(subsubdomain)} : {len(subdomain_files)}")
            data_dict[cls_ind].append(subdomain_files)
        else:
            print(f"{str(subsubdomain)} : {len(subdomain_files)} < {min_num_files}")
    return data_dict


def flatten_data_dict(data_dict: Dict[int, List[List[Path]]]):
    """
    data dict example = {
        0 : [
            [Path,Path,...], # Subdomain files
            [Path,Path,...], # Subdomain files
        ],
        1: [
            [Path, Path, ...], # Subdomain files
            [Path, Path, ...], # Subdomain files
        ],
    }
    """
    utt_list = []
    for cls_ind, cls_domains in data_dict.items():
        for domain_files in cls_domains:
            utt_list.extend(zip(domain_files, [cls_ind] * len(domain_files)))
    return utt_list


def subset_test_data(data_config: Dict[str, int],
                     ext: str = "wav",
                     domain_subset_size: int = 100,
                     random=True) -> List[Tuple[Path, str, str, int]]:
    # Returns List[Tuple[path_to_test_file,domain,subdomain,label]]
    test_data = []
    for subdomain_root in data_config.keys():
        cls_ind = data_config[subdomain_root]
        subdomain_root = Path(subdomain_root)
        sub_subdomain_roots = list(subdomain_root.glob("*/"))
        print(f"{subdomain_root} : sub_subdomain_roots : {len(sub_subdomain_roots)}")
        for sub_subdomain_root in sub_subdomain_roots:
            subdomain_files = list(sub_subdomain_root.glob(f"**/*.{ext}"))
            picked_subdomain_files = None
            if random:
                if len(subdomain_files) > domain_subset_size:
                    picked_subdomain_files = np.random.choice(subdomain_files,
                                                              size=domain_subset_size,
                                                              replace=False)
                else:
                    picked_subdomain_files = subdomain_files[:domain_subset_size]
            else:
                picked_subdomain_files = subdomain_files[:domain_subset_size]
            test_data.extend(
                zip(
                    picked_subdomain_files,
                    # [subdomain_root.name]*len(picked_subdomain_files),
                    # [sub_subdomain_root.name]*len(picked_subdomain_files),
                    [cls_ind] * len(picked_subdomain_files)))
    print(f"Subset of test data : {len(test_data)}")
    return test_data
