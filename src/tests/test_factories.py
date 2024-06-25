import json

import pytest  # noqa: F401
from omegaconf import DictConfig

from main.structure.Factories.ConfigurationFactory import ConfigurationFactory
from main.structure.Factories.FilterFactory import FilterFactory
from main.structure.Factories.PipelineFactory import PipelineFactory
from main.structure.Filters.CreateDatasetFilter import CreateDatasetFilter
from main.structure.Filters.Excel2JSONFilter import Excel2JSONFilter
from main.structure.Filters.ExtendAnnotationFilter import ExtendAnnotationFilter
from main.structure.Filters.MergeNewGeneratedJSONFilesFilter import MergeNewGeneratedJSONFilesFilter
from main.structure.Filters.MergeOriginalJSONFilesFilter import MergeOriginalJSONFilesFilter
from main.structure.Filters.PredictionFilter import PredictionFilter
from main.structure.Filters.PrepareDatasetForPredictionFilter import PrepareDatasetForPredictionFilter
from main.structure.Filters.PrepareDatasetForTrainingFilter import PrepareDatasetForTrainingFilter
from main.structure.Filters.TrainAndEvaluateModelsFilter import TrainAndEvaluateModelsFilter
from main.structure.Pipelines.CreationPipeline import CreationPipeline
from main.structure.Pipelines.TrainingPipeline import TrainingPipeline

configFactory = ConfigurationFactory()
trainingConf = configFactory.__create__("TrainingAllDatasets")
creationConf = configFactory.__create__("AnnotationAndDataset")

with open("src/tests/testContent.json", 'r') as file:
    content = json.load(file)

with open("src/tests/testAnnotationWithoutCodes.json", 'r') as file:
    annotation = json.load(file)


def test_configuration_factory() -> None:
    configFactory = ConfigurationFactory()

    confTraining = configFactory.__create__("TrainingAllDatasets")
    assert isinstance(confTraining, DictConfig)

    confTraining = configFactory.__create__("TrainingOnlyP2GoldenDataset")
    assert isinstance(confTraining, DictConfig)

    confTraining = configFactory.__create__("TrainingOnlyKomootDataset")
    assert isinstance(confTraining, DictConfig)

    confTraining = configFactory.__create__("TrainingKomootTestP2GoldenDatasets")
    assert isinstance(confTraining, DictConfig)

    confTraining = configFactory.__create__("TrainingP2GoldenTestKomootDatasets")
    assert isinstance(confTraining, DictConfig)

    confOnlyAnnotation = configFactory.__create__("OnlyAnnotation")
    assert isinstance(confOnlyAnnotation, DictConfig)

    confOnlyDataset = configFactory.__create__("OnlyDataset")
    assert isinstance(confOnlyDataset, DictConfig)

    confAnnotationAndDataset = configFactory.__create__("AnnotationAndDataset")
    assert isinstance(confAnnotationAndDataset, DictConfig)


def test_pipeline_factory() -> None:
    pipelineFactory = PipelineFactory()

    trainingPipeline = pipelineFactory.__create__("TrainingPipeline", conf=trainingConf)
    assert isinstance(trainingPipeline, TrainingPipeline)

    creationPipeline = pipelineFactory.__create__("CreationPipeline", conf=creationConf, content=content, annotation=annotation)
    assert isinstance(creationPipeline, CreationPipeline)


def test_filter_factory() -> None:
    filterFactory = FilterFactory()

    mergeNewGeneratedJSONFilesFilter = filterFactory.__create__("MergeNewGeneratedJSONFilesFilter", conf=trainingConf)
    assert isinstance(mergeNewGeneratedJSONFilesFilter, MergeNewGeneratedJSONFilesFilter)

    createDatasetFilter = filterFactory.__create__("CreateDatasetFilter", conf=creationConf, content=content)
    assert isinstance(createDatasetFilter, CreateDatasetFilter)

    prepareDatasetForTrainingFilter = filterFactory.__create__("PrepareDatasetForTrainingFilter", conf=trainingConf)
    assert isinstance(prepareDatasetForTrainingFilter, PrepareDatasetForTrainingFilter)

    excel2JSONFilter = filterFactory.__create__("Excel2JSONFilter", conf=trainingConf)
    assert isinstance(excel2JSONFilter, Excel2JSONFilter)

    extendAnnotationFilter = filterFactory.__create__("ExtendAnnotationFilter", annotation=annotation)
    assert isinstance(extendAnnotationFilter, ExtendAnnotationFilter)

    mergeOriginalJSONFilesFilter = filterFactory.__create__("MergeOriginalJSONFilesFilter", conf=trainingConf)
    assert isinstance(mergeOriginalJSONFilesFilter, MergeOriginalJSONFilesFilter)

    predictionFilter = filterFactory.__create__("PredictionFilter", conf=creationConf)
    assert isinstance(predictionFilter, PredictionFilter)

    prepareDatasetForPredictionFilter = filterFactory.__create__("PrepareDatasetForPredictionFilter", annotation=annotation)
    assert isinstance(prepareDatasetForPredictionFilter, PrepareDatasetForPredictionFilter)

    trainAndEvaluateModelsFilter = filterFactory.__create__("TrainAndEvaluateModelsFilter", conf=trainingConf)
    assert isinstance(trainAndEvaluateModelsFilter, TrainAndEvaluateModelsFilter)
