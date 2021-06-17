import yaml
import os
import datetime


class ConfigParser(object):
    def __init__(self, config_dict):
        # convert config dict to a class instance
        for name, value in config_dict.items():
            setattr(self, name, self._wrap(value))

    def _wrap(self, value):
        if isinstance(value, (tuple, list, set, frozenset)):
            return type(value)([self._wrap(v) for v in value])
        else:
            return ConfigParser(value) if isinstance(value, dict) else value

    @classmethod
    def from_yaml(cls, config_fn, mode="train"):
        with open(config_fn) as file:
            config_dict = yaml.full_load(file)

        # in train mode, create dedicated experiment dir (given by experiment_name) inside reports
        if mode == "train":
            if config_dict["experiment_name"] is not None:
                config_dict["TRAINER"]["save_dir"] = os.path.join(
                    config_dict["TRAINER"]["save_dir"], config_dict["experiment_name"]
                )
            else:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                config_dict["TRAINER"]["save_dir"] = os.path.join(
                    config_dict["TRAINER"]["save_dir"], timestamp
                )
            os.makedirs(config_dict["TRAINER"]["save_dir"], exist_ok=True)

        return cls(config_dict)


if __name__ == "__main__":
    CFG = ConfigParser.from_yaml(config_fn="../train_config.yaml")
    print(CFG)
