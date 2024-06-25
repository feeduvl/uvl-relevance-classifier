import json
import os
import random

import pandas as pd
import pytest  # noqa: F401

from main.structure.DataModels import Annotation
from main.structure.Factories.ConfigurationFactory import ConfigurationFactory
from main.structure.Factories.FilterFactory import FilterFactory
from main.tooling.FileManager import cleanup, getPathForNewGeneratedFiles, getPathForOriginalDatasets

configFactory = ConfigurationFactory()
trainingConf = configFactory.__create__("TrainingAllDatasets")
trainingDatasetsSplittedConf = configFactory.__create__("TrainingKomootTestP2GoldenDatasets")
creationConf = configFactory.__create__("AnnotationAndDataset")

with open("src/tests/testContent.json", 'r') as file:
    content = json.load(file)

with open("src/tests/testAnnotationWithoutCodes.json", 'r') as file:
    annotationWithoutCodes = json.load(file)

with open("src/tests/testAnnotationWithCodes.json", 'r') as file:
    annotationWithCodes = json.load(file)


def test_Excel2JSONFilter() -> None:
    expected_lines = ['{"text":"Now I bought an iPhone six running iOS 8.","labels":0}', '{"text":"It installed fine.","labels":0}']
    filterFactory = FilterFactory()
    excel2JSONFilter = filterFactory.__create__("Excel2JSONFilter", conf=trainingConf)
    excel2JSONFilter.__filter__()
    dfOriginal = pd.read_excel(getPathForOriginalDatasets(trainingConf.datasources.original_datasources.original_excel_file_name))
    with open(getPathForNewGeneratedFiles(name=trainingConf.datasources.new_datasources.new_excel2json_file_name), 'r') as file:
        linesNew = file.readlines()

    # test, if the new .json file is correctly created
    assert os.path.exists(getPathForNewGeneratedFiles(name=trainingConf.datasources.new_datasources.new_excel2json_file_name))
    # test, if the new .json file contains correct line format and content
    for index, line in enumerate(expected_lines):
        assert linesNew[index].strip() == line
    # test, if the number of lines is the same
    assert len(dfOriginal) == len(linesNew)


def test_MergeOriginalJSONFilesFilter() -> None:
    expected_lines = ['{"text": "A little bit too much self promotion in the app and by email.", "labels": 1}', '{"text": "Other than that its solid, works as expected.", "labels": 1}']
    filterFactory = FilterFactory()
    mergeOriginalJSONFilesFilter = filterFactory.__create__("MergeOriginalJSONFilesFilter", conf=trainingConf)
    mergeOriginalJSONFilesFilter.__filter__()

    with open(getPathForNewGeneratedFiles(name=trainingConf.datasources.new_datasources.new_combined_original_jsons_file_name), 'r') as file:
        linesNew = file.readlines()

    # test, if the new .json file is correctly created
    assert os.path.exists(getPathForNewGeneratedFiles(name=trainingConf.datasources.new_datasources.new_combined_original_jsons_file_name))

    # test, if the new .json file contains correct line format and content
    for index, line in enumerate(expected_lines):
        assert linesNew[index].strip() == line

    # test, if the new .json file contains the correct number of lines (codes)
    assert 2199 == len(linesNew)


def test_MergeNewGeneratedJSONFilesFilter() -> None:
    expected_lines = ['{"text":"A little bit too much self promotion in the app and by email.","labels":1}', '{"text":"Other than that its solid, works as expected.","labels":1}']
    filterFactory = FilterFactory()
    mergeNewGeneratedJSONFilesFilter = filterFactory.__create__("MergeNewGeneratedJSONFilesFilter", conf=trainingConf)
    mergeNewGeneratedJSONFilesFilter.__filter__()

    with open(getPathForNewGeneratedFiles(name=trainingConf.datasources.new_datasources.new_combined_datasets_file_name), 'r') as file:
        newCombinedLines = file.readlines()

    with open(getPathForNewGeneratedFiles(name=trainingConf.datasources.new_datasources.new_excel2json_file_name), 'r') as file:
        newExcelLines = file.readlines()

    # test, if the new .json file is correctly created
    assert os.path.exists(getPathForNewGeneratedFiles(name=trainingConf.datasources.new_datasources.new_combined_datasets_file_name))

    # test, if the new concatenation in the new .json file is correct (for the 2. part we have
    # to use the expected_lines, since there are whitespaces behind the key in the dict (no
    # influence for the code))
    for index, line in enumerate(newExcelLines):
        assert newCombinedLines[index] == line

    for index, line in enumerate(expected_lines, start=len(newExcelLines)):
        assert newCombinedLines[index].strip() == line


def test_PrepareDatasetForTrainingFilter() -> None:
    filterFactory = FilterFactory()
    prepareDatasetForTrainingFilter = filterFactory.__create__("PrepareDatasetForTrainingFilter", conf=trainingConf)
    prepareDatasetForTrainingFilterSplittedConf = filterFactory.__create__("PrepareDatasetForTrainingFilter", conf=trainingDatasetsSplittedConf)

    dfTrainTest = prepareDatasetForTrainingFilter.__filter__()
    dfTrainTestTuple = prepareDatasetForTrainingFilterSplittedConf.__filter__()

    # test if dfTrainTest and dfTrainTestTuple have the correct type
    assert isinstance(dfTrainTest, pd.DataFrame)
    assert isinstance(dfTrainTestTuple, tuple)
    assert isinstance(dfTrainTestTuple[0], pd.DataFrame)
    assert isinstance(dfTrainTestTuple[1], pd.DataFrame)


def test_PrepareDatasetForPredictionFilter() -> None:
    annotationMapped = Annotation(
        uploaded_at=annotationWithoutCodes["uploaded_at"],
        last_updated=annotationWithoutCodes["last_updated"],
        name=annotationWithoutCodes["name"],
        dataset=annotationWithoutCodes["dataset"],
        tores=annotationWithoutCodes.get("tores", []),
        show_recommendationtore=annotationWithoutCodes.get("show_recommendationtore", False),
        docs=annotationWithoutCodes["docs"],
        tokens=annotationWithoutCodes["tokens"],
        codes=annotationWithoutCodes["codes"],
        tore_relationships=annotationWithoutCodes["tore_relationships"]
    )

    filterFactory = FilterFactory()
    prepareDatasetForPredictionFilter = filterFactory.__create__("PrepareDatasetForPredictionFilter", annotation=annotationMapped)
    sentencesForPrediction = prepareDatasetForPredictionFilter.__filter__()

    # test if sentencesForPrediction has the correct type (List[str])
    assert isinstance(sentencesForPrediction, list)
    assert all(isinstance(sentence, str) for sentence in sentencesForPrediction)

    # test, if sentencesForPrediction has the same length as the original tokenlist
    assert len(sentencesForPrediction) == len(annotationMapped['tokens'])

    # test, if sentencesForPrediction has the same content as the names of the original tokenlist
    for index, sentence in enumerate(sentencesForPrediction):
        assert sentence == annotationMapped['tokens'][index]['name']


def test_ExtendAnnotationFilter() -> None:
    annotationMapped = Annotation(
        uploaded_at=annotationWithoutCodes["uploaded_at"],
        last_updated=annotationWithoutCodes["last_updated"],
        name=annotationWithoutCodes["name"],
        dataset=annotationWithoutCodes["dataset"],
        tores=annotationWithoutCodes.get("tores", []),
        show_recommendationtore=annotationWithoutCodes.get("show_recommendationtore", False),
        docs=annotationWithoutCodes["docs"],
        tokens=annotationWithoutCodes["tokens"],
        codes=annotationWithoutCodes["codes"],
        tore_relationships=annotationWithoutCodes["tore_relationships"]
    )

    filterFactory = FilterFactory()
    prepareDatasetForPredictionFilter = filterFactory.__create__("PrepareDatasetForPredictionFilter", annotation=annotationMapped)

    sentences = prepareDatasetForPredictionFilter.__filter__()
    labels = [random.choice(["Informative", "Non-Informative"]) for _ in range(len(sentences))]
    print("sentences: ", sentences)

    extendAnnotationFilter = filterFactory.__create__("ExtendAnnotationFilter", annotation=annotationMapped)
    extendedAnnotation = extendAnnotationFilter.__filter__((sentences, labels))

    # test, if the codelist length of extendedAnnotation is the same as the tokenlist length of annotationMapped
    assert len(extendedAnnotation['codes']) == len(annotationMapped['tokens'])

    # test, if all tokens in the annotation got the value "1" for the field "num_tore_codes"
    for token in extendedAnnotation['tokens']:
        assert token['num_tore_codes'] == 1


def test_CreateDatasetFilter() -> None:
    annotationMapped = Annotation(
        uploaded_at=annotationWithCodes["uploaded_at"],
        last_updated=annotationWithCodes["last_updated"],
        name=annotationWithCodes["name"],
        dataset=annotationWithCodes["dataset"],
        tores=annotationWithCodes.get("tores", []),
        show_recommendationtore=annotationWithCodes.get("show_recommendationtore", False),
        docs=annotationWithCodes["docs"],
        tokens=annotationWithCodes["tokens"],
        codes=annotationWithCodes["codes"],
        tore_relationships=annotationWithCodes["tore_relationships"]
    )

    filterFactory = FilterFactory()
    createDatasetFilter = filterFactory.__create__("CreateDatasetFilter", conf=creationConf, content=content)
    annotation, newGeneratedDataset = createDatasetFilter.__filter__(annotationMapped)

    originalDatasetDocs = annotationMapped['docs']
    newGeneratedDatasetDocs = newGeneratedDataset['documents']

    # test, if newGeneratedDataset has the correct doc IDs
    for index, doc in enumerate(newGeneratedDatasetDocs):
        assert doc['id'] == originalDatasetDocs[index]['name']

    doc1InformativeText = "Very useful app for planning trips. This makes it a pain to work with on mobile devices."
    doc2InformativeText = "Starting a tour, the track disappears from the map."
    doc3InformativeText = "It has never let me down or taken me to a dead end. It's very reliable and has only gotten better with the growing community of hikers and cyclists (at least in Germany). Quick tip: if you're using S8 (or similar) turn off Edge Lighting in display settings while using Wake-up Display during navigation."

    # test, if newGeneratedDataset has the correct doc only informative text
    assert newGeneratedDatasetDocs[0]['text'] == doc1InformativeText
    assert newGeneratedDatasetDocs[1]['text'] == doc2InformativeText
    assert newGeneratedDatasetDocs[2]['text'] == doc3InformativeText

    # test, if newGeneratedDataset has the correct size
    assert newGeneratedDataset['size'] == len(originalDatasetDocs)

    cleanup()
