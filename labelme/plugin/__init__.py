import os.path as osp

# from .register import *
# from .pixel_cd import *
from .object_cd import object_cd

here = osp.dirname(osp.abspath(__file__))


class Plugin(object):

    def __init__(self, **kargs):
        super().__init__()
        output_configs = kargs.pop('output', None)
        plugin_configs = kargs.pop('plugins', None)

        self.output_configs = output_configs
        self.plugins = self.get_plugins(plugin_configs=plugin_configs)

    def get_plugins(self, plugin_configs=None):
        plugins = []

        if not plugin_configs:
            return plugins

        if not isinstance(plugin_configs, list):
            plugin_configs = [plugin_configs]

        for cfg in plugin_configs:
            input_from = cfg.pop('from', -1)
            fusion = cfg.pop("fusion", None)
            plugin = eval(cfg['name'])(**cfg['args'])
            plugins.append({
                "from": input_from,
                "plugin": plugin,
                "fusion": fusion,
            })

        return plugins

    def make_output(self, outputs):
        if not self.output_configs:
            return outputs[-1]
        choices = self.output_configs['from']
        if not isinstance(choices, list):
            choices = [choices]
        output = [outputs[c] for c in choices]
        return output

    def __call__(self, x):
        # import pdb; pdb.set_trace()
        outputs = [x]
        for p in self.plugins:
            x_in = [outputs[_] for _ in p['from']]
            if p['fusion'] == None: x_in = x_in[0]
            outputs.append(p['plugin'](x_in))
        output = self.make_output(outputs)

        return output

if __name__ == "__main__":
    import yaml
    import cv2

    with open('./plugin.yaml', 'r') as f:
        config = yaml.safe_load(f)

    img = cv2.imread('/Users/shinian/proj/data/stb/train/A/6.tif', -1)
    img_1 = cv2.imread('/Users/shinian/proj/data/stb/train/B/6.tif', -1)

    model = Plugin(**config['plugin'])
    x = model([img, img_1])
    print(x)


