# Large Replay Parallel Dataset

This repository contains code for experiments described in paper.


#### Dependencies
Install dependencies by running `pip install -r requirements.txt`

#### Data

Before running train, one should acquire datasets:
* LRPD dataset using [link]()
* ASVSpoof2017 dataset using provided [script](utils/load_asv17.py) by running `python utils/load_asv17.py OUTPUT_FOLDER` or accessing it on [official web page](https://datashare.ed.ac.uk/handle/10283/3055)
* [MUSAN](http://www.openslr.org/17/),[DECASE 2017 Task3 Acoustic Scenes](http://dcase.community/challenge2017/task-sound-event-detection-in-real-life-audio),[DEMAND](https://deepai.org/dataset/demand) datasets as noise datasets

And setup pathes to dataset roots in `data.yml`:
```
lrpd_root: /PATH/TO/LRPD
asv17_root: /PATH/TO/ASVSpoof2017
noise_roots: 
    - /PATH/TO/MUSAN/ [optional]
    - /PATH/TO/DEMAND/ [optional]
    - /PATH/TO/DECASE/ [optional]
    ...
```

### ADD:
* Description of available model architectures in **model_config.json**

#### Training

* Export current path to environment
```bash
export PYTHONPATH=$(pwd)
```

To run training refer to:  `python3 train.py --help`:
```
usage: train.py [-h] [--train_config TRAIN_CONFIG] [--model_config MODEL_CONFIG] [--gpus GPUS] [--exp_dir EXP_DIR] task dataset_setup

positional arguments:
  task                  antispoofing or device_detector
  dataset_setup         Dataset setup in form 'lrpd_office,lrpd_aparts,asv17_train'

optional arguments:
  -h, --help            show this help message and exit
  --train_config TRAIN_CONFIG
                        Path to train config serialized into JSON (default: /media/ssdraid0cgpu01/home/iiakovlev/new-pipeline/audio-pipelines-pytorch/configs/common/train_config.json)
  --model_config MODEL_CONFIG
                        Path to model config serialized into JSON (default: /media/ssdraid0cgpu01/home/iiakovlev/new-pipeline/audio-pipelines-pytorch/configs/common/model_config.json)
  --gpus GPUS           IDs of GPUs to train on. For example : 0,1 (default: 0)
  --exp_dir EXP_DIR     Path to experiment folder (default: None)
```

For example: `python3 train.py antispoofing lrpd_office,lrpd_aparts,asv17_train` will run training of antispoofing detector using LRPD-office, LRPD-aparts and ASVSpoof2017 train part as training data