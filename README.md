# Stellar mass and radius estimation using Artificial Intelligence
This repository includes the codes and data to reproduce the experiments detailed in the following paper:

```bibtex
@article{Moya2021,
  author  = {A. Moya and R.~J. {L\'opez-Sastre}},
  title   = {Stellar mass and radius estimation using Artificial Intelligence},
  journal = {Astronomy & Astrophysics},
  pages = {1--10},
  volume = {Published online},
  number = {},
  issn = {0004-6361},
  doi = {https://doi.org/10.1051/0004-6361/202142930},
  year	= {2022}
}
```
## Data sample
The data sample detailed in the paper can be found in the folder `data`.


## Usage
Clone this repository. Then, to reproduce the results of the paper, we provide the python3 script [mr_estimation_train_test.py](mr_estimation_train_test.py). 

Simply run the script with the following command:
```bash
python3 mr_estimation_train_test.py
```
By default, the script trains and evaluates all the AI models detailed in the paper for the stellar mass.

To generate the AI models for the stellar radius, uncomment [line 204 of mr_estimation_train_test.py](https://github.com/gramuah/ai4mr/blob/d7dde2eb7991f43af188a9c220cc718d21b4560d/mr_estimation_train_test.py#L204) (and comment line 203).

### Reproducing the plots included in the paper
For reproducing the plots included in the paper we provide script [mr_estimation_paper_plots.py](mr_estimation_paper_plots.py).

### Generalization experiment
The generalization experiment can be reproduced running the script [mr_estimation_generalization.py](mr_estimation_generalization.py).

### Features influence
For the discussion about the features influence provided in the paper, the results can be reproduced running the script [mr_estimation_features_influence.py](mr_estimation_features_influence.py).

