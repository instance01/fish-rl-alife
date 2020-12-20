import copy
import json


class Config:
    def __init__(self):
        with open('simulations.json', 'r') as f:
            self.cfg = json.load(f)

    def get_cfg(self, cfg_id):
        base_cfg = self.cfg[self.cfg[cfg_id].get('base_cfg', cfg_id)]
        cfg = copy.deepcopy(base_cfg)
        if cfg_id not in self.cfg:
            raise Exception(
                'Error: Key %s does not exist in simulations.json.' % cfg_id
            )
        # We support one level for now.
        for k in self.cfg[cfg_id].keys():
            if k == 'base_cfg':
                continue
            cfg[k].update(self.cfg[cfg_id][k])
        cfg['cfg_id'] = cfg_id
        return cfg


if __name__ == '__main__':
    cfg = Config()
    print('\n'.join(cfg.cfg.keys()))
