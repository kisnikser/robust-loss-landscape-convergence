import itertools
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy
import scipy.stats as sps

from tqdm.auto import tqdm

from src.params_directions import create_random_direction
from src.params_directions import init_from_params
from src.params_directions import calc_sum_models

from src.utils import create_losses_func

def calc_grid_loss(model, calc_losses_func, coef_grid, direction_norm):
    """
        grid: 2d from [-1, 1]*[-1, 1]
    """
    result = {}
    direction1 = create_random_direction(model, external_factor=direction_norm)
    direction2 = create_random_direction(model, external_factor=direction_norm)
    target_model = None
    for coef1, coef2 in tqdm(coef_grid):
        target_add = [p1*coef1 + p2*coef2 for p1, p2 in zip(direction1, direction2)]
        target_add_model = copy.deepcopy(model)
        init_from_params(target_add_model, target_add)

        del target_model
        target_model = calc_sum_models(model, target_add_model, 1, 1)
        losses = calc_losses_func(target_model)
        result[(coef1, coef2)] = losses
    return result

class LossVisualizer:
    def __init__(self, visual_type = 'random_2', grid_step = 0.1, direction_norm = 1):
        """
            grid_step in [0, 1]
        """
        self.visual_type = visual_type
        self.grid_step = grid_step
        self.direction_norm = direction_norm

    def initialize_grid_loss(self, model, calc_losses_func):
        """
            calc_loss_func(model) == loss
        """
        coef_grid = itertools.product(np.arange(-1, 1+self.grid_step, step = self.grid_step), 
                                           np.arange(-1, 1+self.grid_step, step = self.grid_step))
        self.grid_loss = calc_grid_loss(model, calc_losses_func, coef_grid, self.direction_norm)

    def _set_xy_grid(self, x_grid_bounds, y_grid_bounds):
        xs = np.arange(-1, 1+self.grid_step, step = self.grid_step)
        ys = xs
        xs = np.array([x for x in xs if x<=x_grid_bounds[1] and x >= x_grid_bounds[0]])
        ys = np.array([y for y in ys if y<=y_grid_bounds[1] and y >= y_grid_bounds[0]])
        return xs, ys

    def visualize_diff(self, size1, size2, 
                       x_grid_bounds = (-1, 1), y_grid_bounds = (-1, 1), 
                       diff_type = None, bounds = None, distrib_params = None):
        """
           type: None | abs | square | realative | relative_abs | relative_squar | square_dot_normal | abs_dot_normal
           bounds: None | (-min, max)
        """
        assert bounds is None or bounds[0] < bounds[1]
        grid_loss = self.grid_loss
        
        xs, ys = self._set_xy_grid(x_grid_bounds, y_grid_bounds) 
        xgrid, ygrid = np.meshgrid(xs, ys)
        zgrid1 = np.array([[np.mean(grid_loss[(x, y)][:size1]) for x in xs] for y in ys])
        zgrid2 = np.array([[np.mean(grid_loss[(x, y)][:size2]) for x in xs] for y in ys])
        zgrid = zgrid2 - zgrid1

        best_loss = np.round(min(np.min(zgrid1), np.min(zgrid2)), 4)
        
        relative = False
        if diff_type is not None and diff_type[0:2] == 're':
            zgrid /= zgrid2
            relative = True
            
        if diff_type is not None and diff_type.find('normal') != -1:
            pdf = lambda x,y : np.prod(sps.norm(**distrib_params).pdf(np.array([x, y])))
            zgrid *= np.array([[pdf(x, y) for x in xs] for y in ys])
            
        if diff_type is None:
            title = '$\mathcal{L}_{s_2} - \mathcal{L}_{s_1}$'
            if relative:
                title += '/ \mathcal{L}_{s2}'
        elif diff_type == 'abs' or diff_type == 'abs_dot_normal' or diff_type == 'relative_abs':
            title = '$|\mathcal{L}_{s_2} - \mathcal{L}_{s_1}|$'
            zgrid = np.abs(zgrid)
            if relative:
                title += '$/ \mathcal{L}_{s2}$'
        elif diff_type == 'square' or diff_type == 'square_dot_normal' or diff_type == 'relative_square':
            title = '$(\mathcal{L}_{s_2} - \mathcal{L}_{s_1})^2$'
            zgrid = np.square(zgrid)
            if relative:
                title += '$/ \mathcal{L}_{s2}^2$'

        dot_distrib_flag = False
        if diff_type is not None and diff_type.find('normal') != -1:
            title += '$p(\mathbf{w})$'
            dot_distrib_flag = True
            
        title += f'; $s_1 = {size1}, s_2 = {size2}$'
        title += ', optimal loss :' + f' {best_loss}'
        if dot_distrib_flag:
            title += f'\n distribution params: {distrib_params}'
            
        def bounds_func(x):
            if bounds is None:
                return x
            x = min(x, bounds[1])
            x = max(x, bounds[0])
            return x
        zgrid = np.array([[bounds_func(x) for x in r] for r in zgrid])
        
            
        fig = plt.figure(figsize=(6, 6))
        ax_3d = Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(ax_3d)
        ax_3d.set(
            title = title
        )
        if bounds is not None:
            ax_3d.set_zlim3d(bounds[0], bounds[1])
        surf = ax_3d.plot_surface(xgrid, ygrid, zgrid, linewidth=0, antialiased=False, cmap=cm.coolwarm, alpha = 1.0)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax_3d.view_init(40, 20)
        plt.show() 

    def visualize(self, size = None, 
                  x_grid_bounds = (-1, 1), 
                  y_grid_bounds = (-1, 1), 
                  z_grid_bounds = (-float('inf'), float('inf'))):
        grid_loss = self.grid_loss

        xs, ys = self._set_xy_grid(x_grid_bounds, y_grid_bounds) 
        xgrid, ygrid = np.meshgrid(xs, ys)

        def bounds_func(x):
            bounds = z_grid_bounds
            if bounds is None:
                return x
            x = min(x, bounds[1])
            x = max(x, bounds[0])
            return x
        
        if size is None:
            zgrid = np.array([[bounds_func(np.mean(grid_loss[(x, y)])) for x in xs] for y in ys])
            title = '$\mathcal{L}_{s};  s = -1$'
        else:
            zgrid = np.array([[bounds_func(np.mean(grid_loss[(x, y)][:size])) for x in xs] for y in ys])
            title = '$\mathcal{L}_{s}; $' + f's = {size}'

        max_loss = np.max(np.array([[(np.mean(grid_loss[(x, y)])) for x in xs] for y in ys]))
        best_loss = np.min(np.array([[(np.mean(grid_loss[(x, y)])) for x in xs] for y in ys]))

        if z_grid_bounds[0] == -float('inf'):
            z_grid_bounds[0] = best_loss
        
        if z_grid_bounds[1] == float('inf'):
            z_grid_bounds[1] = max_loss

        best_loss = np.round(best_loss, 3)
        title +=  f' optimal loss: {best_loss}'

        fig = plt.figure(figsize=(6, 6))
        ax_3d = Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(ax_3d)
        ax_3d.set(
            title = title
        )
        surf = ax_3d.plot_surface(xgrid, 
                ygrid, 
                zgrid, 
                linewidth=0, 
                antialiased=False, 
                cmap=cm.coolwarm, 
                alpha = 1.0)

        ax_3d.set_zlim3d(z_grid_bounds[0], z_grid_bounds[1])

        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax_3d.view_init(40, 20)
        
        plt.show()
