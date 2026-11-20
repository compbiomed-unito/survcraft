# Notebook tutorials

The repository includes notebooks for interactive exploration:

- [Basic usage](https://github.com/compbiomed-unito/survcraft/blob/main/tutorials/Basic%20usage.ipynb)
- [Simulation](https://github.com/compbiomed-unito/survcraft/blob/main/tutorials/Simulation.ipynb): sample predicted curves, use fixed linear maps and FractalNoise baselines, and assess generated outcomes.
- [Simulation optimization with Optuna](https://github.com/compbiomed-unito/survcraft/blob/main/tutorials/Simulation%20optimization%20with%20Optuna.ipynb): fine-tune dataset properties and check the selected generator on fresh samples.
- [Implementing custom layers](https://github.com/compbiomed-unito/survcraft/blob/main/tutorials/Implementing%20custom%20layers.ipynb)

Clone the repository and install the tutorial dependencies:

```bash
git clone https://github.com/compbiomed-unito/survcraft.git
cd survcraft
pip install -e ".[tutorials]"
jupyter notebook tutorials/
```

Both simulation tutorials also require `survmetr`; the optimization notebook
additionally requires `optuna`. Their setup sections include installation instructions.

The documentation build links to the notebooks rather than executing them.
