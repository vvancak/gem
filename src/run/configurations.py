import run.print_headers as ph
import typing as t
import json
import os

DS_CONFIG = "ds_config.json"
EM_CONFIG = "em_config.json"
EV_CONFIG = "ev_config.json"


class Configuration():
    def __init__(self, base_path: str):
        self._base_path = base_path

    def _get_config(self, filename: str, path: str, default_path: str = None) -> t.Dict:
        path = f"{path}/{filename}"
        default_path = f"{default_path}/{filename}"

        if not os.path.exists(path):
            print(f"{ph.WARN} Dataset-Specific config path {path} does not exist!")

            if default_path:
                print(f"{ph.INFO} Loading configuration from default path: {default_path}")
                with open(default_path, "r") as file:
                    return json.load(file)

            print(f"{ph.ERROR} No config loaded!")
            return {}
        else:
            print(f"{ph.INFO} Loading configuration from: {path}")
            with open(path, "r") as file:
                return json.load(file)

    def em_config(self, dataset: str):
        ds_specific_path = f"{self._base_path}/{dataset}"
        return self._get_config(EM_CONFIG, ds_specific_path, self._base_path)

    def ev_config(self, dataset: str):
        ds_specific_path = f"{self._base_path}/{dataset}"
        return self._get_config(EV_CONFIG, ds_specific_path, self._base_path)

    def ds_config(self):
        return self._get_config(DS_CONFIG, self._base_path)
