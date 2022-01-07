# Scalable Inference in SDEs by Direct Matching of the Fokker–Planck–Kolmogorov Equation

This repository is the official implementation of the methods in the publication
* Solin, A., Tamir, E., and Verma, P. (2021) **Scalable Inference in SDEs by Direct Matching of the Fokker–Planck–Kolmogorov Equation**. In *Advances in Neural Information Processing Systems 35 (NeurIPS)*. [[arXiv]](https://arxiv.org/abs/2110.15739)

In the paper, we advocate alternative solution concepts to stochastic differential equation (SDE) models in machine learning, where simulation-based techniques such as variants of stochastic Runge–Kutta are currently the *de facto* approach. These methods are convenient, general-purpose, and used with parametric and non-parametric models, and neural SDEs. Yet, stochastic Runge–Kutta relies on the use of sampling schemes that can be inefficient in high dimensions. We address this issue by revisiting the classical SDE literature and derive direct approximations to the (typically intractable) Fokker–Planck–Kolmogorov equation by matching moments. The codebase in this repository includes the building blocks for the figures and code for the experiments in the paper.

## Python environment

The code should be run using python 3.6. If you are already using python 3.6, dependencies can be installed using the requirements file
```bash
pip install -r requirements.txt
```
Alternatively, conda virtual environment can be created using the `environment.yml` file

```bash
conda env create -f environment.yml
conda activate scalable-sde
```

## MOCAP experiment

The MOCAP experiment specific code is in `experiments/mocap`. To prepare the data for the experiment, place
the Mocap matlab data file `mocap35.mat` to folder `[base_folder]/data/mocap_data`, where `base_folder` is to be given as input to the
training and test scripts. We use the preprocessed MOCAP data from https://github.com/cagatayyildiz/ODE2VAE.

In order to run the MOCAP training, run
 ```bash
python experiments/mocap/walking_tf_train.py [-base_folder BASE_FOLDER] [-task TASK] [-decoder_dist DECODER_DIST]
                  [-model_name MODEL_NAME] [-prior_model_name PRIOR_MODEL_NAME] [-vae_name VAE_NAME]
                  [--dt DT] [--latent_dim LATENT_DIM] [--context_dim CONTEXT_DIM]
                  [--epochs EPOCHS] [--start_len START_LEN]
````


For testing a trained MOCAP model, run
 ```bash
python experiments/mocap/walking_tf_test.py [-base_folder BASE_FOLDER] [-task TASK] [-decoder_dist DECODER_DIST]
                  [-model_name MODEL_NAME] [-vae_name VAE_NAME] [--dt DT]
                  [--latent_dim LATENT_DIM] [--context_dim CONTEXT_DIM] [--start_len START_LEN]
```

See the train and test scripts for further documentation of their input arguments.
To modify the codebase for some other flat dataset (VAE implementation doesn't support image data),
modify the utility function `get_data` in `experiments/mocap/walking_tf_functions.py` to output another dataset class.



### Alternative SDE Approximations
You can run the MOCAP experiment with any new SDE approximator, as long as it inherits from the class
`SDEApprox` in `src/sde_tf/sde_approx/sde_approx.py`.

## Rotating MNIST

The code for the rotating MNIST experiment is available in `experiments/mnist`. In order to run the experiment:

```bash
cd experiments/mnist/
python main.py
```

All the experiment related parameters are present in `config.py` from where they can be modified. By default the output folder is `experiments/mnist/output` where the trained models and inference plots are saved.

## Notebooks

The code used to generate `Figure 3` and `Figure 4` of the paper is available in the jupyter notebook, `/notebooks/`.

## Citation
If you use the code in this repository for your research, please cite the paper as follows:
```bibtex
@inproceedings{solin2021,
  title={Scalable Inference in SDEs by Direct Matching of the {F}okker--{P}lanck--{K}olmogorov Equation},
  author={Solin, Arno and Tamir, Ella and Verma, Prakhar},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year={2021}
}
```

## Contributing

For all correspondence, please contact arno.solin@aalto.fi, ella.tamir@aalto.fi, or prakhar.verma@aalto.fi .

## License

This software is provided under the [MIT license](LICENSE).
