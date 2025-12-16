from src.calc_delta import DeltaCalculator
from src.visualize import LossVisualizer
from src.utils import create_losses_func

import matplotlib.pyplot as plt
import numpy as np

import copy

class DeltaCalcVisualizer:
    def __init__(self, model, loader, criterion, external_factor = 1.0):
        self.delta_calc = DeltaCalculator(model, loader, criterion, external_factor)
        pass
    def compare_params(self,
            mode, 
            params, 
            target_param_key, 
            target_param_grid, 
            num_samples,
            begin):
        params = copy.deepcopy(params)
        target_param_to_deltas = {}
        for target_param in target_param_grid:
            params[target_param_key] = target_param
            deltas = self.delta_calc.calc_deltas(mode, params, num_samples = num_samples)
            target_param_to_deltas[target_param] = deltas

        fig, axs = plt.subplots(figsize = (12, 4), nrows = 1, ncols = 2)
        fig.suptitle(f'params: {params}')
        for target_param, deltas in target_param_to_deltas.items():
            axs[0].plot(deltas, label = target_param)
            mult_coef = np.arange(1, len(deltas)+1)
            if params['estim_func'] == 'square':
                mult_coef = mult_coef**2
            axs[1].plot(deltas * mult_coef, label = target_param)

        if params['estim_func'] == 'abs':
            ylabels = ['$\int_{w}|\mathcal{L}_{k+1}(w) - \mathcal{L}_k|(w)$', '$k\Delta_k$']
        elif params['estim_func'] == 'square':
            ylabels = ['$\int_{w}|\mathcal{L}_{k+1}(w) - \mathcal{L}_k|^2(w)$', '$k\Delta_k$']
        else:
            ylabels = ['$(\mathcal{L}_{k+1} - \mathcal{L}_k)$', '$k(\mathcal{L}_{k+1} - \mathcal{L}_k)*k$']

        axs[0].set(
            title = f'Compare {target_param_key}',
            xlabel = 'k',
            ylabel = ylabels[0],
            xlim = [begin, len(deltas)],
            ylim = [min(deltas[begin:]), max(deltas[begin:])*1.2]
        )

        axs[1].set(
            title = f'Compare {target_param_key}',
            xlabel = 'k',
            ylabel = ylabels[1],
        )

        axs[0].legend(title = target_param_key)
        axs[1].legend(title = target_param_key)
        
        plt.show()

    def compare_samples_num(self,
            mode,
            params,
            num_samples_grid,
            begin = 0
            ):

        max_samples_num = max(num_samples_grid)
        diff_lists = self.delta_calc.calc_diff_lists(mode, params, num_samples = max_samples_num)
        if params['estim_func'] == 'square':
            diff_lists = diff_lists**2
        elif params['estim_func'] == 'abs':
            diff_lists = np.abs(diff_lists)
        
        cummean_diff_lists = np.cumsum(diff_lists, axis = 0) / np.arange(1, len(diff_lists) + 1).reshape(-1, 1)
        num_samples_to_deltas = {b:cummean_diff_lists[b-1] for b in num_samples_grid}

        fig, axs = plt.subplots(figsize = (12, 4), nrows = 1, ncols = 2)
        fig.suptitle(f'params: {params}')
        for num_samples, deltas in num_samples_to_deltas.items():
            axs[0].plot(deltas, label = num_samples)
            mult_coef = np.arange(1, len(deltas)+1)
            if params['estim_func'] == 'square':
                mult_coef = mult_coef**2
            axs[1].plot(deltas * mult_coef, label = num_samples)

        if params['estim_func'] == 'abs':
            ylabels = ['$\int_{w}|\mathcal{L}_{k+1}(w) - \mathcal{L}_k|(w)$', '$k\Delta_k$']
        elif params['estim_func'] == 'square':
            ylabels = ['$\int_{w}|\mathcal{L}_{k+1}(w) - \mathcal{L}_k|^2(w)$', '$k\Delta_k$']
        else:
            ylabels = ['$(\mathcal{L}_{k+1} - \mathcal{L}_k)$', '$k(\mathcal{L}_{k+1} - \mathcal{L}_k)*k$']

        axs[0].set(
            title = f'Compare num_samples',
            xlabel = 'k',
            ylabel = ylabels[0],
            xlim = [begin, len(deltas)],
            ylim = [min(deltas[begin:]), max(deltas[begin:])*1.2]
        )

        axs[1].set(
            title = f'Compare num_samples',
            xlabel = 'k',
            ylabel = ylabels[1],
        )

        axs[0].legend(title = 'num_samples')
        axs[1].legend(title = 'num_samples')
        
        plt.show()
