# HolStep baselines

This is the code used for the baseline models described in the paper ["HolStep: a Machine Learning Dataset for Higher-Order Logic Theorem Proving"](http://cl-informatik.uibk.ac.at/cek/holstep/ckfccs-holstep-submitted.pdf).
These models are very basic and unoptimized, and we expect it should be easy to improve upon them.

The HolStep dataset may be download [here](http://cl-informatik.uibk.ac.at/cek/holstep/).


## Usage

After having downloaded the dataset to e.g. `~/Downloads/holstep`, run `main.py`:

```sh
python main.py \
--model_name=cnn_2x_siamese \
--task_name=conditioned_classification \
--logdir=experiments/cnn_2x_siamese_experiment \
--source_dir=/Users/fchollet/Downloads/holstep
```

`model_name` is the name of a model (see which models are available in `conditioned_classification_models.py` and `unconditioned_classification_models.py`).
`task_name` may be either `conditioned_classification` or `unconditioned_classification`.
