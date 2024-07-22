from typing import Dict, cast

from omegaconf import DictConfig

from main.connector.ForeignComponentRelevanceClassifierRestConnectorRequester import ForeignComponentRelevanceClassifierRestConnectorRequester
from main.structure.DataModels import Annotation
from main.structure.Factories.ConfigurationFactory import ConfigurationFactory
from main.structure.Factories.PipelineFactory import PipelineFactory
from main.tooling.AnnotationMapper import AnnotationMapper
from main.tooling.Logger import logging_setup

logger = logging_setup(__name__)


class RelevanceClassifier():
    """
        Description: This is the behavior class, that uses the classes in the structure package to delegate the program.
    """

    def __init__(self) -> None:
        self.foreignComponentRequester = ForeignComponentRelevanceClassifierRestConnectorRequester()
        self.pipelineFactory = PipelineFactory()
        self.configurationFactory = ConfigurationFactory()

    def startTrainingPipeline(self, conf: DictConfig) -> None:
        """
            Description:
                This method delegates the training.
            Args:
                DictConfig: The configuration, that specifies, which training shall be executed
            Returns:
                None: At the end of the training pipeline, a model and evaluation metrics are uploaded to MLflow. This is technical code, which
                is handled in the MLflowHandler
        """

        trainingPipeline = self.pipelineFactory.__create__("TrainingPipeline", conf=conf)
        trainingPipeline.__process__()

    def startCreationPipeline(self, content: Dict[str, str]) -> str:
        """
            Description:
                This method delegates the creation.
            Args:
                Dict[str, str]: The selected configuration from the user (comes from the ri-visualization service)
            Returns:
                str: A message, that outlines what was created
        """

        try:
            conf = self.configurationFactory.__create__(content["params"]["relevance_classification_conf"])  # type: ignore[index]
        except KeyError as e:
            logger.error(f"KeyError: Missing key in content: {e}")

        if conf.name == "creation_annotation_config":
            logger.info("-------Relevance Prediction with Annotation Creation requested-------")

            annotation = self.initializeAndGetAnnotation(content)
            creationPipeline = self.pipelineFactory.__create__("CreationPipeline", conf=conf, content=content, annotation=annotation)
            finishedAnnotation = creationPipeline.__process__()

            self.foreignComponentRequester.storeAnnotationRequest(finishedAnnotation)
            return "New annotation successfully created!"

        elif conf.name == "creation_dataset_config":
            logger.info("-------Relevance Prediction with Dataset Creation requested-------")

            sentenceTokenizationEnabledForAnnotation = True
            annotation = self.foreignComponentRequester.tokenizeAnnotationRequest(content["dataset"], sentenceTokenizationEnabledForAnnotation)
            creationPipeline = self.pipelineFactory.__create__("CreationPipeline", conf=conf, content=content, annotation=annotation)
            finishedAnnotation, newGeneratedDataset = creationPipeline.__process__()

            if newGeneratedDataset:
                self.foreignComponentRequester.storeDatasetRequest(newGeneratedDataset)
                return "New dataset successfully created!"

            return "New dataset contains no informative sentences and is therefor not created!"

        elif conf.name == "creation_annotation_and_dataset_config":
            logger.info("-------Relevance Prediction with Annotation and Dataset Creation requested-------")

            annotation = self.initializeAndGetAnnotation(content)
            creationPipeline = self.pipelineFactory.__create__("CreationPipeline", conf=conf, content=content, annotation=annotation)
            finishedAnnotation, newGeneratedDataset = creationPipeline.__process__()

            self.foreignComponentRequester.storeAnnotationRequest(finishedAnnotation)

            if newGeneratedDataset:
                self.foreignComponentRequester.storeDatasetRequest(newGeneratedDataset)
                return "New dataset and annotation successfully created!"

            return "New dataset contains no informative sentences and is therefor not created! New annotation successfully created!"

        else:
            raise ValueError(f"Invalid Conf Name: {conf.name}")

    def initializeAndGetAnnotation(self, content: Dict[str, str]) -> Annotation:
        """
            Description:
                This method uses the foreignComponentRequester to get a new initialized sentence-based annotation.
            Args:
                Dict[str, str]: The selected configuration from the user (comes from the ri-visualization service), that
                contains the needed dataset and annotation name for the annotation creation
            Returns:
                Annotation: The new initialized annotation
        """

        annotationName = content["params"]["new_annotation_name"]  # type: ignore[index]
        datasetName = content["dataset"]["name"]  # type: ignore[index]
        sentenceTokenizationEnabledForAnnotation = True

        self.foreignComponentRequester.initializeAnnotationRequest(datasetName, annotationName, sentenceTokenizationEnabledForAnnotation)
        annotation = cast(Annotation, self.foreignComponentRequester.getAnnotationRequest(annotationName))

        return annotation

    def mapRelevantSentences2WordBasedAnnotation(self, wordBasedAnnotation: str, sentenceBasedAnnotation: str) -> None:
        """
            Description:
                This method uses the foreignComponentRequester and delegates the annotation mapping, which is based
                on a word-based and sentence-based annotation with the same dataset as a base.
            Args:
                str: The name of the word-based annotation
                str: The name of the sentence-based annotation
            Returns:
                None: A mapped word-based annotation is saved in the database via the foreignComponentRequester
        """

        wordBasedAnnotation = cast(Annotation, self.foreignComponentRequester.getAnnotationRequest(wordBasedAnnotation))
        sentenceBasedAnnotation = cast(Annotation, self.foreignComponentRequester.getAnnotationRequest(sentenceBasedAnnotation))

        annotationMapper = AnnotationMapper(wordBasedAnnotation, sentenceBasedAnnotation)
        wordBasedAnnotation = annotationMapper.mapRelevantSentences2WordBasedAnnotation()

        self.foreignComponentRequester.storeAnnotationRequest(wordBasedAnnotation)
