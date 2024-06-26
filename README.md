<h1 align="center" style="margin-top: 0px;">uvl-relevance-classifier</h1>

# Describtion
This microservice implements the automatic relevance classification of online feedback. For the classifier, the [pre-trained "bert-base-uncased" model](https://huggingface.co/google-bert/bert-base-uncased), which is fine-tuned for the task, is used.
Five runs are executed, were models are fine-tuned based on different datasets and dataset combinations. The models and the evaluation of these models are stored in MLflow.

Two different datasets are used:
- The P2-Golden dataset (consists of 1242 app review sentences) from [^fn1]
- A new manual labeled dataset, that consists of 2199 app review sentences for the app "Komoot"

The five dataset methods are:
- Training and testing only with the P2-Golden dataset
- Training and testing only with the Komoot dataset
- Training and testing by combining the P2-Golden and Komoot datasets
- Training with the P2-Golden and testing with the Komoot dataset
- Training with the Komoot and testing with the P2-Golden dataset

The training is processed via 5-fold cross-validation.

The service provides two methods:
- get the status of the service
- automatically create an annotation and a new dataset with only informative app review sentences

# Requirements
- running MLflow server
- existing model in MLflow and adjusted experiment parameters depending on your model choice in the jupyter notebook with the name: "ComponentRelevanceClassifierServiceSetup"

## Hardware

## Software
- >=Python 3.11

# Getting Started as containerized microservice

```sh
docker build -t <CONTAINER_NAME> -f "./Dockerfile" --build-arg mlflow_tracking_username=XXXXXX --build-arg mlflow_tracking_password=XXXXXX --build-arg mlflow_tracking_uri=https://mlflow-uvl.ifi.uni-heidelberg.de .
```
→ replace XXXXXX with the appropriate credentials

# Getting Started as local testing
## 1. Clone the repository

```sh
git clone https://github.com/feeduvl/uvl-relevance-classifier.git
cd uvl-relevance-classifier
```

## 2. Create a virtual environment (optional)

### venv
...

### conda

**1. conda installation:**
[Installation of miniconda](https://docs.anaconda.com/free/miniconda/miniconda-install)

**2. conda activation:**
```sh
conda create -n rc python=3.11
conda activate rc
```

**3. Setting up local MLflow environment parameters:**

```sh
cd $CONDA_PREFIX
mkdir -p ./etc/conda/activate.d
mkdir -p ./etc/conda/deactivate.d
echo \#\!/bin/bash >> ./etc/conda/activate.d/env_vars.sh
echo "export MLFLOW_TRACKING_USERNAME=" >> ./etc/conda/activate.d/env_vars.sh
echo "export MLFLOW_TRACKING_PASSWORD=" >> ./etc/conda/activate.d/env_vars.sh
echo "export MLFLOW_TRACKING_URI='http://127.0.0.1:5000'" >> ./etc/conda/activate.d/env_vars.sh

echo \#\!/bin/bash >> ./etc/conda/deactivate.d/env_vars.sh
echo "unset MLFLOW_TRACKING_USERNAME" >> ./etc/conda/deactivate.d/env_vars.sh
echo "unset MLFLOW_TRACKING_PASSWORD" >> ./etc/conda/deactivate.d/env_vars.sh
echo "unset MLFLOW_TRACKING_URI" >> ./etc/conda/deactivate.d/env_vars.sh

conda deactivate
conda activate rc
```

**Start local MLflow server:**
```sh
cd <THE_BASE_DIRECTORY_FOR_MLFLOW>
mlflow server
```

## 3. Setting up the development environment

```sh
cd <THE_BASE_DIRECTORY_FOR_THE_uvl-relevance-classifier_SERVICE>
pip install -e .
```

## 4. Train the relevance classification model and load the existing model

**If there is no existing relevance classification model on MLflow, train the models:** <br>
Start the jupyter notebook with the name: "ComponentRelevanceClassifierServiceTraining"

**Load the existing model from MLflow, you want to use for the relevance classification:** <br>
Start the jupyter notebook with the name: "ComponentRelevanceClassifierServiceSetup"
→ in this notebook, you have to adjust the experiment parameters depending on your model choice

## 5. Start the uvl-relevance-classifier service
```sh
./start.sh
```

## References
[^fn1]:van Vliet, M., Groen, E., Dalpiaz, F., Brinkkemper, S.: Crowd-annotation results: Identifying and classifying user requirements in online feedback (2020), https://doi.org/10.5281/zenodo.3754721, Zenodo
