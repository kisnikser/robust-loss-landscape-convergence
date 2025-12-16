# Robust Convergence of Loss Landscapes through Distributional Averaging

[![paper](https://img.shields.io/badge/paper-preprint-blue.svg)](https://doi.org/10.1134/S1064562424601987)
[![slides](https://img.shields.io/badge/slides-presentation-red.svg)](slides/main.pdf)

This is the official implementation of the paper **Robust Convergence of Loss Landscapes through Distributional Averaging** by [Nikita Kiselev](https://kisnikser.github.io/), [Vladislav Meshkov](https://github.com/VseMeshkov) and [Andrey Grabovoy](https://scholar.google.com/citations?user=ZtI9pgsAAAAJ&hl=ru&oi=sra).

<img alt="overview" width=700 src="paper/losses_difference_monte_carlo.svg">

> **Abstract:** _Understanding how a neural network’s loss landscape evolves with dataset size is essential for identifying sufficient training data. Prior analyses of this problem have typically been local, focusing on second-order expansions around a single optimum and bounding convergence through Hessian properties. While such studies clarify convergence rates, they provide only a pointwise view of stability. In this paper, we extend the framework to a distributional paradigm. Instead of analyzing convergence at one optimum, we evaluate it in expectation over a parameter distribution. This approach captures how entire neighborhoods of the loss landscape stabilize as additional samples are added. We focus on Gaussian distributions centered at local minima and employ Monte Carlo sampling to estimate convergence in practice. Theoretically, we show that distributional convergence exhibits the same asymptotic rate as the local case, while offering a more robust picture of stability. Empirical studies on image classification tasks confirm these predictions and highlight how architectural choices such as normalization, dropout, and network depth influence convergence. Our results broaden local convergence analyses into a distributional setting, providing stronger guarantees and practical tools for characterizing dataset sufficiency._

## Abstract

TODO

## Citation

If you find our work helpful, please cite us.

```BibTeX
@article{citekey,
    title={Title},
    author={Name Surname, Name Surname (consultant), Name Surname (advisor)},
    year={2025}
}
```

## Licence

Our project is MIT licensed. See [LICENSE](LICENSE) for details.
