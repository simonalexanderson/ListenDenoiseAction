import os
import numpy as np

from argparse import ArgumentParser, Namespace
from jsmin import jsmin
import json
import yaml
from pytorch_lightning import Trainer
#from misc.shared import DATA_DIR




def get_hparams():
    parser = ArgumentParser()
    parser.add_argument("dataset_root")
    parser.add_argument("hparams_file")
    parser = Trainer.add_argparse_args(parser)
    default_params = parser.parse_args()

    parser2 = ArgumentParser()
    parser2.add_argument("dataset_root")
    parser2.add_argument("hparams_file")
    override_params, unknown = parser2.parse_known_args()


    conf_name = os.path.basename(override_params.hparams_file)
    if override_params.hparams_file.endswith(".json"):
        hparams_json = json.loads(jsmin(open(override_params.hparams_file).read()))
    elif override_params.hparams_file.endswith(".yaml"):
        hparams_json = yaml.full_load(open(override_params.hparams_file))
    #hparams_json["dataset_root"] = str(data_dir)

    params = vars(default_params)
    params.update(hparams_json)
    params.update(vars(override_params))

    hparams = Namespace(**params)

    return hparams, conf_name
