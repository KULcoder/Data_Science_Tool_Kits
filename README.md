# Data_Science_Tool_Kits
**DS_Tool_Kit_Proposal**

Clearly, there are many libraries fulfilling almost every all functionalities of data science techniques. However, for the purpose of experiment and clarity, a self made data science tool kit might be useful.

**Remark**: This repo is also my experiment on working with *copilot*.

## Specifications

The whole projects is write in `python`.

And all methods input & output types are designed based by `NumPy` and `Pandas`

Restrictions (Errors Raised if wrong case) are implemented

## Structure

Center File `ds_kit.py` : link all other files.

Other functionalities are divided by parts of their purpose (also with a centered file):

### Obtaining Data

`data.py`

- Useful data API, library, data base....

- Parse methods.

### EDA

#### Visualization

Different graphs...

### Data Preprocesses

`data_preprocess.py`

#### Transformation

- one-hot-encoding
- different types of standardization

#### Correction

- ML based missing data imputation

#### Dataloader



### Models

`models.py`

#### Linear Models



#### Model Optimizer

- Stochastic Gradient Descent
- Adam

#### Model Logger



### Evaluation 

`metrics.py`

