# 1p-frac (1-parameter Fractal as Data)

This repository contains the code for generating data with 1p-frac. In 1p-frac, we control the distribution of shape variations using the parameter Δ. Even with very small shape variations, such as Δ=0.1, it achieves pre-training effects close to real data.
<p align="center"> <img src="../figure/fig_liep-1.png" width="90%"/> </p>

## Requirements

* Python 3.x (tested with 3.7)

The code for generating 1p-frac is the same as that used for pre-training.

## Running the Code
To generate 1p-frac data, first specify the hyperparameters in `generate.sh`.

    BASE_SAVEDIR="/path/to/savedir"  # directory to save 1p-frac data
    SIGMA=4.0  # parameter to control shape complexity
    DELTA=0.1  # parameter to control shape variance
    SAMPLE=1000  # Number of samples from the shape distribution controlled by delta

Then, run the code as follows:

    cd 1p-frac/1p-frac_generator
    bash generate.sh

## Dataset Sample
As an example, when adding shape variations in the range of Δ to a given fractal, the distribution of shape changes will look as follows based on Δ:

| Δ  | Sample num | Pre-trained weights | Dataset link |
|----|------------|---------------------|--------------|
| 0.01 | 1k        | TBD                 | TBD          |
| 0.05 | 1k        | TBD                 | TBD          |
| 0.1  | 1k        | TBD                 | TBD          |
| 0.1  | 21k       | TBD                 | TBD          |
