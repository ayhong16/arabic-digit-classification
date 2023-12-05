from matplotlib.colors import Normalize, LinearSegmentedColormap, to_hex



def _map_values(start_value, end_value):
    return [(0.0, start_value, start_value), (1.0, end_value, end_value)]


class GradientPlotter:
    def __init__(self, start_color, end_color, maxvalue):
        self.norm = Normalize(vmin=0, vmax=maxvalue)
        self.cdict = {'red':   _map_values(start_color[0], end_color[0]),
                      'green': _map_values(start_color[1], end_color[1]),
                      'blue':  _map_values(start_color[2], end_color[2])}
        self.cmap = LinearSegmentedColormap('custom_cmap', self.cdict)

    def __call__(self, value):
        rgba = self.cmap(self.norm(value))
        rgb = tuple(rgba[i] for i in range(3))  # Select the first three elements and convert to tuple
        return to_hex(rgb)