# Stellar mass and radius estimation using Artificial Intelligence
This repository includes the codes and data to reproduce the experiments detailed in the following paper:

```bibtex
@article{Moya2021,
  author  = {A. Moya and R.~J. {L\'opez-Sastre}},
  title   = {Stellar mass and radius estimation using Artificial Intelligence},
  journal = {Astronomy & Astrophysics},
  pages = {--},
  volume = {Under review},
  number = {},
  issn = {0004-6361},
  doi = {},
  year	= {2021}
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
