import collections
import itertools
import sys
from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import datasets.dataset_dict
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from main.tooling.FileManager import cleanup, getPathForNewGeneratedFiles
from main.tooling.Logger import logging_setup

logger = logging_setup(__name__)

TRAIN_CSV_FILENAME_FOR_MLFLOW_UPLOAD = "train_dataset.csv"
TEST_CSV_FILENAME_FOR_MLFLOW_UPLOAD = "test_dataset.csv"
CONFUSION_MATRIX_PNG_FILENAME_FOR_MLFLOW_UPLOAD = "confusion_matrix.png"
LABELS = ['Non-Informative', 'Informative']


@contextmanager
def config_mlflow(cfg: DictConfig) -> Iterator[mlflow.ActiveRun]:
    print("cfg Type", type(cfg))
    experiment = mlflow.get_experiment_by_name(cfg.project.experiment_name)
    logger.debug(f"Experiment: {experiment}")

    nested = False
    if mlflow.active_run():
        nested = True

    with mlflow.start_run(
        run_name=cfg.project.run_name,
        experiment_id=experiment.experiment_id,
        nested=nested
    ) as current_run:
        try:
            mlflow.autolog(silent=True, log_models=False)
            log_config(cfg)
            yield current_run
        except KeyboardInterrupt:
            logger.info("Keyboard interupt received")
            cleanup()
            sys.exit()
        except Exception as e:
            logger.error(e)
            cleanup()
            sys.exit()


def log_config(cfg: DictConfig) -> None:
    config = flatten(
        cast(Dict[str, str], OmegaConf.to_container(cfg, resolve=True)),
        separator=".",
    )
    mlflow.log_params(config)


def uploadDatasetToMLflow(df: pd.DataFrame, jsonFileName: str) -> None:

    logger.info("-------Upload dataset to MLflow-------")

    csvFileName = jsonFileName.replace(".json", ".csv")

    df.to_csv(str(getPathForNewGeneratedFiles(name=csvFileName)), index=False)
    logger.debug(f"CSV file saved as: {str(getPathForNewGeneratedFiles(name=jsonFileName))}")

    log_mlflow_artifacts(getPathForNewGeneratedFiles(name=csvFileName))


def createTrainTestFileForMLFlowUpload(foldNumber: int, datasetTrainTest: datasets.dataset_dict.DatasetDict) -> None:

    logger.info("-------Creating Train and Test File for MLflow upload-------")

    datasetTrain = datasetTrainTest["train"]
    datasetTest = datasetTrainTest["test"]

    train_df = pd.DataFrame(datasetTrain)

    test_df = pd.DataFrame(datasetTest)

    trainFileName = f"Iteration_{str(foldNumber)}_{TRAIN_CSV_FILENAME_FOR_MLFLOW_UPLOAD}"
    testFileName = f"Iteration_{str(foldNumber)}_{TEST_CSV_FILENAME_FOR_MLFLOW_UPLOAD}"

    train_df.to_csv(str(getPathForNewGeneratedFiles(name=trainFileName)), index=False)
    logger.debug(f"Train CSV file saved as: {trainFileName}")

    test_df.to_csv(str(getPathForNewGeneratedFiles(name=testFileName)), index=False)
    logger.debug(f"Test CSV file saved as: {testFileName}")

    log_mlflow_artifacts(getPathForNewGeneratedFiles(name=trainFileName))
    log_mlflow_artifacts(getPathForNewGeneratedFiles(name=testFileName))


def createConfusionMatrixPngForMLFlowUpload(foldNumber: int, evaluationResults: dict, total: bool) -> None:

    logger.info("-------Creating Confusion Matrix for MLflow Upload-------")

    if total:
        confusionMatrixName = f"Total_{CONFUSION_MATRIX_PNG_FILENAME_FOR_MLFLOW_UPLOAD}"
    else:
        confusionMatrixName = f"Iteration_{str(foldNumber)}_{CONFUSION_MATRIX_PNG_FILENAME_FOR_MLFLOW_UPLOAD}"

    cm = np.array(evaluationResults['eval_confusion_matrix'])

    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    plt.imshow(cm_percent, interpolation='nearest', cmap=plt.cm.Purples, vmin=0, vmax=1)
    plt.title('Confusion Matrix for Relevance Classification')
    plt.colorbar()
    tick_marks = np.arange(len(LABELS))
    plt.xticks(tick_marks, LABELS)
    plt.yticks(tick_marks, LABELS)

    fmt = '.2f'  # Format as percentage
    thresh = cm_percent.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm_percent[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm_percent[i, j] > thresh else "black")

    plt.ylabel('True Label', fontweight='bold')
    plt.xlabel('Predicted Label', fontweight='bold')
    plt.tight_layout()
    plt.savefig(getPathForNewGeneratedFiles(name=confusionMatrixName))
    plt.close()

    logger.debug(f"Confusion Matrix saved as: {confusionMatrixName}")

    log_mlflow_artifacts(getPathForNewGeneratedFiles(name=confusionMatrixName))


def log_mlflow_artifacts(
    artifact_paths: Path
) -> None:
    mlflow.log_artifact(artifact_paths)


def computeExperimentMetricsForMLFlowUpload(metricsList: List[Dict]) -> None:

    logger.info("-------Computing experiment metrics for MLflow Upload-------")

    metricsDict: Dict[str, List[float]] = {}
    for metrics in metricsList:
        for key, value in metrics.items():
            if key in ['eval_accuracy', 'eval_precision_label_0', 'eval_recall_label_0', 'eval_f1_label_0', 'eval_precision_label_1', 'eval_recall_label_1', 'eval_f1_label_1']:
                if key not in metricsDict:
                    metricsDict[key] = []
                metricsDict[key].append(value)

    metricsSummary: Dict[str, float] = {}
    for key, values in metricsDict.items():
        metricsSummary[f'MAX_{key}'] = np.max(values)
        metricsSummary[f'MIN_{key}'] = np.min(values)
        metricsSummary[f'MEAN_{key}'] = np.mean(values)

    for key, value in metricsSummary.items():
        mlflow.log_metric(key, value)

    totalConfusionMatrix = np.zeros((2, 2), dtype=int)

    for metrics in metricsList:
        eval_cm = metrics['eval_confusion_matrix']
        totalConfusionMatrix += np.array(eval_cm)

    totalConfusionMatrixDict = {}
    totalConfusionMatrixDict["eval_confusion_matrix"] = totalConfusionMatrix

    createConfusionMatrixPngForMLFlowUpload(0, totalConfusionMatrixDict, True)

    logger.debug(f"Experiment metrics uploaded to MLflow: {metricsSummary}")


def flatten(
    dictionary: MutableMapping[str, Any],
    parent_key: Optional[str] = None,
    separator: str = "_",
) -> Dict[str, Any]:
    """
        Description:
            Turn a nested dictionary into a flattened dictionary
        Args:
            MutableMapping[str, Any]: The dictionary to flatten
            None: The string to prepend to dictionary's keys
            str: The string used to separate flattened keys
        Returns: 
            A flattened dictionary
    """

    items: List[Tuple[str, Any]] = []
    for key, value in dictionary.items():
        new_key = str(parent_key) + separator + key if parent_key else key
        if isinstance(value, collections.abc.MutableMapping):
            items.extend(flatten(value, new_key, separator).items())
        elif isinstance(value, list):
            for k, v in enumerate(value):
                items.extend(flatten({str(k): v}, new_key).items())
        else:
            items.append((new_key, value))
    return dict(items)
