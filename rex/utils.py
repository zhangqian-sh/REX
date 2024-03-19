import os
import time

import h5py
import numpy as np
import yaml
from absl import logging
from ml_collections import ConfigDict

# logging


class CustomPythonFormatter(logging.PythonFormatter):
    def format(self, record):
        time_tuple = time.localtime(record.created)
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time_tuple)
        info_level = logging.converter.get_initial_for_level(record.levelno)
        prefix = f"{info_level} {time_str} {record.filename}:{record.lineno}] "
        return prefix + super(logging.PythonFormatter, self).format(record)


def setup_logger(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    logging.get_absl_handler().use_absl_log_file("log", log_dir)
    logging.get_absl_handler().setFormatter(CustomPythonFormatter())


# config


def load_config_from_file(config_path: str):
    """
    Load config from yaml file, turn it into a ml_collections.ConfigDict
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    cfg = ConfigDict(config)
    return cfg

def setup_output_folder(cfg: dict):
    cfg.io.train_data_path = os.path.join(cfg.io.root_folder, cfg.io.train_data_path)
    if isinstance(cfg.io.test_data_path, str):
        cfg.io.test_data_path = os.path.join(cfg.io.root_folder, cfg.io.test_data_path)
    elif isinstance(cfg.io.test_data_path, dict) or isinstance(cfg.io.test_data_path, ConfigDict):
        for key in cfg.io.test_data_path:
            cfg.io.test_data_path[key] = os.path.join(cfg.io.root_folder, cfg.io.test_data_path[key])
    else:
        raise ValueError(f"cfg.io.test_data_path should be either str or dict, got {type(cfg.io.test_data_path)}")
    cfg.io.result_folder = os.path.join(cfg.io.root_folder, cfg.io.result_folder, cfg.task.name)
    cfg.test.model_path = os.path.join(cfg.io.result_folder, cfg.test.model_path)
    cfg.io.result_folder = cfg.io.result_folder
    cfg.io.debug_folder = os.path.join(cfg.io.result_folder, "debug")
    cfg.io.model_folder = os.path.join(cfg.io.result_folder, "model")
    cfg.io.output_folder = os.path.join(cfg.io.result_folder, "output")
    os.makedirs(cfg.io.debug_folder, exist_ok=True)
    os.makedirs(cfg.io.model_folder, exist_ok=True)
    os.makedirs(cfg.io.output_folder, exist_ok=True)
    return cfg


# load and save data


def recursively_save_dict_contents_to_group(h5f: h5py.File, path: str, d: dict):
    for key, item in d.items():
        if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes)):
            h5f[path + key] = item
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5f, path + key + "/", item)
        else:
            raise ValueError("Cannot save %s type" % type(item))


def load_data_from_file(data_path: str):
    with h5py.File(data_path, "r") as f:
        data = recursively_load_dict_contents_from_group(f, "/")
    return data


def recursively_load_dict_contents_from_group(h5f: h5py.File, path: str):
    ans = {}
    for key, item in h5f[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item[()]
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5f, path + key + "/")
    return ans


# print tree


def h5_tree(h5f: h5py.File, pre=""):
    items = len(h5f)
    for key, h5f in h5f.items():
        items -= 1
        if items == 0:
            # the last item
            if type(h5f) == h5py._hl.group.Group:
                print(pre + "└── " + key)
                h5_tree(h5f, pre + "    ")
            else:
                print(pre + "└── " + key + f" ({h5f.shape})")
        else:
            if type(h5f) == h5py._hl.group.Group:
                print(pre + "├── " + key)
                h5_tree(h5f, pre + "│   ")
            else:
                print(pre + "├── " + key + f" ({h5f.shape})")


def np_tree(np_dict: dict, pre: str = ""):
    items = len(np_dict)
    output = ""
    for key, item in np_dict.items():
        items -= 1
        if items == 0:
            # the last item
            if isinstance(item, np.ndarray):
                output += f"{pre}└── {key} ({item.shape})\n"
            else:
                output += f"{pre}└── {key}\n"
                output += np_tree(item, pre + "    ")
        else:
            if isinstance(item, np.ndarray):
                output += f"{pre}├── {key} ({item.shape})\n"
            else:
                output += f"{pre}├── {key}\n"
                output += np_tree(item, pre + "│   ")
    return output
