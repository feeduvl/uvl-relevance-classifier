import os
import shutil
from pathlib import Path

from main.tooling.Logger import logging_setup

logger = logging_setup(__name__)

ORIGINAL_FILES_PATH = "src/data/"
TEMP_FILES_PATH = "src/data/temp/"
MODEL_PATH = "src/main/models/"
CONFIG_FILES_PATH = "src/main/configs/"


def getPathForOriginalDatasets(name: str) -> Path:
    return Path(ORIGINAL_FILES_PATH + name)


def getPathForNewGeneratedFiles(name: str) -> Path:

    if not Path(TEMP_FILES_PATH).exists():
        Path(TEMP_FILES_PATH).mkdir()

    return Path(TEMP_FILES_PATH + name)


def getConfigPath(name: str) -> str:
    return CONFIG_FILES_PATH + name


def getModelPath(name: str) -> Path:
    if not Path(MODEL_PATH).exists():
        Path(MODEL_PATH).mkdir()

    return MODEL_PATH + name  # type: ignore


def cleanup() -> None:
    if os.path.isdir(TEMP_FILES_PATH):
        logger.info("-------Cleanup new generated Files (temp directory)-------")
        shutil.rmtree(TEMP_FILES_PATH)
    else:
        logger.info("-------No new generated Files to clean (temp directory)-------")

    if os.path.isdir(MODEL_PATH):
        logger.info("-------Cleanup models (model directory)-------")
        shutil.rmtree(MODEL_PATH)
    else:
        logger.info("-------No new trained model to clean (model directory)-------")
