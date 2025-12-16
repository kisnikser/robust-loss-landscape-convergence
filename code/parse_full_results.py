import pandas as pd
import glob
import json

def get_full_try_exps_dataframe(path = 'results'):
    rows = []
    for file in glob.glob(f'{path}/**/hc*/**/*.json', recursive=True):
        with open(file, 'r') as file:
            d = json.load(file)
            d['sigma'] = d['delta_vis_params']['mode_params']['sigma']
            d['num_samples'] = d['delta_vis_params']['num_samples']
            del d['delta_vis_params']
            rows.append(d)

    data = pd.DataFrame(rows)
    return data
