#!/usr/bin/env python3
import argparse
import os
import subprocess
from pathlib import Path
from urllib.parse import urlparse


URLS_TO_LOAD = [
    "https://datashare.ed.ac.uk/bitstream/handle/10283/3055/protocol_V2.zip",
    "https://datashare.ed.ac.uk/bitstream/handle/10283/3055/ASVspoof2017_V2_dev.zip",
    "https://datashare.ed.ac.uk/bitstream/handle/10283/3055/ASVspoof2017_V2_train.zip",
    "https://datashare.ed.ac.uk/bitstream/handle/10283/3055/ASVspoof2017_V2_eval.zip",
]


def run_sh(command: str, cwd: str = None):
    after_strip = map(lambda s: s.strip(), command.split(' '))
    after_filter = filter(None, after_strip)
    after_filter = list(after_filter)

    process = subprocess.Popen(after_filter,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               cwd=cwd)

    stdout, stderr = process.communicate()
    return stdout.decode('utf-8'), stderr.decode('utf-8')


def load_artefacts(folder):
    cwd = os.getcwd()
    folder = Path(folder).resolve()
    folder.mkdir(exist_ok=True, parents=True)
    os.chdir(folder)
    for url in URLS_TO_LOAD:
        fn = Path(urlparse(url).path).name
        wget_command = f"wget -O {fn} {url}"
        out, err = run_sh(wget_command)
        unzip_command = f"unzip {fn}"
        out, err = run_sh(unzip_command)
        rm_command = f"rm {fn}"
        out, err = run_sh(unzip_command)
    os.chdir(cwd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("target_folder",
                        type=str,
                        help="Folder to where load ASVSpoof 2017 dataset")

    args = parser.parse_args()
    load_artefacts(args.target_folder)
