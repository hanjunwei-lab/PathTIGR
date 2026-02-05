# PathTIGR: A pathway topology-informed graph representation learning framework for immunotherapy response prediction
-----------------------------------------------------------------
Code by **Xiangmei Li** at Harbin Medical University.
This repository contains source code and data for **PathTIGR** 

## 1. Introduction

**PathTIGR** is a Pathway Topology-Informed Graph Representation learning framework that systematically integrates biological pathway network topology knowledge with genome variation information for immunotherapy response prediction.
PathTIGR employs a three-component design: (1) pathway graph encoder with multi-head attention embeding pathway topology knowledge and cancer genomic variants to pathway activity representation; (2) transformer module capturing pathway regulatory dependencies, and (3) multilayer perceptron synthesizing pathway-level representations to predict immunotherapy response. This architecture enables PathTIGR to capture complex molecular interactions underlying immunotherapy response while maintaining biological interpretability.

## 2. Design of PathTIGR

![alt text](image/workflow.jpg "Design of PathTIGR")

Figure 1: Overall architecture of PathTIGR

## 3. Installation

* Python (version 3.8.16)
    * **PathTIGR** relies on [Python (version 3.9.13)](https://www.python.org/downloads/release/python-390/) environments.
* Anaconda
    * Comprehensive installation instructions for Anaconda are available at: https://docs.conda.io/projects/conda/en/latest/user-guide/install/.

    * Following the installation of Anaconda, a virtual environment designated as PathTIGR can be created and the requisite dependencies can be installed from the ``GAE_environment.yml`` configuration file by executing the following command:
    ```
    conda env create -f GAE_environment.yml
    ```
* PyTorch
    PyTorch requires separate installation tailored to the specific hardware configuration. The appropriate installation command can be obtained from https://pytorch.org/.


## 4. Usage

This study trained **PathTIGR** models for different immunotherapy inhibitors separately. All the codes and data required to execute **PathTIGR** are provided in this GitHub repository. Please make sure to replace the input data path in the code with your own storage location.

### 4.1. Code
- The complete code for **PathTIGR** is located at folder ``PathTIGR/``, with the details of each file as follows:

| File                              | Description                                                                   |
|------------------------------------|------------------------------------------------------------------------|
| DataPreprocessing.py                             | Data pre-processing                            |
| CustomDataset.py                           | Custom dataloader|
| EarlyStopping.py                           | The function of early stopping mechanism                            |
| GenerateModelInput.py                           | The function of generate model input data                            |
| GAE.py | The function of the graph autoencoder model                                       |
| model.py | The function of the PathTIGR model                                       |
| train_PathTIGR.ipynb | Training the PathTIGR model for predicting the cancer immunotherapy response                                       |
| Comparative algorithm | Implementation of the Comparative Algorithm                                       |

- Users may reproduce the **PathTIGR** model by following the implementation provided in *train_PathTIGR.ipynb*, or retrain the model using custom datasets. The principal trainable parameters are specified as follows::


    * ``--GAE_epochs``:  The number of training iterations in GAE model.
    * ``--hidden1_dim``:  The dimension of the neurons in the first hidden layer.
    * ``--hidden2_dim``:  The dimension of the neurons in the second hidden layer.
    * ``--global_epo``:  The number of total training iterations in PathTIGR model.
    * ``--learning_rate``:  The learning_rate of training the PathTIGR model.
    * ``--batch_size``:  The number of patients for each batch.
    * ``--num_heads``:  The number of attention heads.
    * ``--early_stopping_patience``: The number of patience in early stopping mechanism
    * ``--dropout_prob``: The dropout possibility for PathTIGR model
    

### 4.2. Data
- The datasets used to train **PathTIGR** are located at folder ``data/``:

| File                              | Description                                                                   |
|------------------------------------|------------------------------------------------------------------------|
| PathwayXML                             | The pathway data from the KEGG database in XML format                            |
| SKCM144/110/60/30                           | The original data of four immunotherapy datasets                             |
| ./SKCM144/pre-processing | The pre-processing data of four immunotherapy datasets                                       |


## 5. Interpretation of the **PathTIGR** model

To elucidate the biological mechanisms underlying **PathTIGR**’s predictions and identify key molecular determinants of immunotherapy response, we employed a dual **attention-based** importance scoring framework that integrates gene-level and pathway-level interpretability.

