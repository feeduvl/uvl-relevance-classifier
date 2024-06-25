from typing import Dict, List, cast

import requests
from requests.exceptions import ConnectionError

from main.structure.DataModels import Annotation, Dataset
from main.tooling.Logger import logging_setup

logger = logging_setup(__name__)


ANNOTATION_GET_ENDPOINT = "http://localhost:9684/hitec/repository/concepts/annotation/name/"
ANNOTATION_POST_ENDPOINT = "http://localhost:9684/hitec/repository/concepts/store/annotation/"
ANNOTATION_INITIALIZE_ENDPOINT = 'http://localhost:9709/hitec/orchestration/concepts/annotationinit/'
ANNOTATION_TOKENIZE_ENDPOINT = "http://localhost:9665/hitec/annotation/tokenize/"
DATASET_POST_ENDPOINT = "http://localhost:9684/hitec/repository/concepts/store/dataset/"


class ForeignComponentRelevanceClassifierRestConnectorRequester():
    """
        Description: REST calls to foreign services.
    """

    def __init__(self) -> None:
        pass

    def getAnnotationRequest(self, annotationName: str) -> Annotation:
        """
            Description:
                This method sends a get request to the uvl-storage service to get an existing annotation.
            Args:
                str: The annotation name
            Returns:
                Annotation: The requested annotation
        """

        logger.info(f"-------Get created Annotation with Name {annotationName}-------")

        try:
            response = requests.get(f"{ANNOTATION_GET_ENDPOINT}{annotationName}")
        except ConnectionError as e:
            print(f"ConnectionError: Failed to connect to the uvl-storage service: {e}")

        return cast(Annotation, response.json())

    def storeAnnotationRequest(self, annotation: Annotation) -> int:
        """
            Description:
                This method sends a post request to the uvl-storage service to store an finished annotation.
            Args:
                Annotation: The finishhed annotation, which has to be stored
            Returns:
                int: The status of the post request
        """

        logger.info(f"-------Store extended Annotation: {annotation}-------")

        try:
            response = requests.post(ANNOTATION_POST_ENDPOINT, json=annotation)
        except ConnectionError as e:
            print(f"ConnectionError: Failed to connect to the uvl-storage service: {e}")

        return response.status_code

    def initializeAnnotationRequest(self, datasetName: str, annotationName: str, sentenceTokenizationEnabledForAnnotation: bool) -> int:
        """
            Description:
                This method sends a post request to the uvl-orchestration service to initialize a new annotation.
            Args:
                str: The dataset name
                str: The annotation name
                bool: What kind of tokenization
            Returns:
                int: The status of the post request
        """

        logger.info(f"-------Initialize annotation {annotationName} of dataset {datasetName}-------")

        annotation = {"name": annotationName, "dataset": datasetName, "sentenceTokenizationEnabledForAnnotation": sentenceTokenizationEnabledForAnnotation}

        try:
            response = requests.post(
                ANNOTATION_INITIALIZE_ENDPOINT,
                json=annotation,
            )
        except ConnectionError as e:
            raise ConnectionError(f"ConnectionError: Failed to connect to the uvl-orchestration service: {e}")

        return response.status_code

    def tokenizeAnnotationRequest(self, dataset: List[Dict], sentenceTokenizationEnabledForAnnotation: bool) -> Annotation:
        """
            Description:
                This method sends a post request to the uvl-annotation service to tokenize a dataset.
            Args:
                List[Dict]: The dataset, which has to be tokenized
                bool: What kind of tokenization
            Returns:
                Annotation: A tokenized annotation
        """

        logger.info("-------Tokenize dataset-------")

        datasetForTokenization = {"dataset": dataset, "sentenceTokenizationEnabledForAnnotation": sentenceTokenizationEnabledForAnnotation}

        try:
            response = requests.post(
                ANNOTATION_TOKENIZE_ENDPOINT,
                json=datasetForTokenization,
            )
        except ConnectionError as e:
            print(f"ConnectionError: Failed to connect to the uvl-annotation service: {e}")

        return cast(Annotation, response.json())

    def storeDatasetRequest(self, dataset: Dataset) -> int:
        """
            Description:
                This method sends a post request to the uvl-storage service to store a dataset.
            Args:
                Dataset: The dataset, which has to be stored
            Returns:
                int: The status of the post request
        """

        logger.info(f"-------Store new generated Dataset with the Name: {dataset}-------")

        try:
            response = requests.post(DATASET_POST_ENDPOINT, json=dataset)
        except ConnectionError as e:
            print(f"ConnectionError: Failed to connect to the uvl-storage service: {e}")

        return response.status_code
