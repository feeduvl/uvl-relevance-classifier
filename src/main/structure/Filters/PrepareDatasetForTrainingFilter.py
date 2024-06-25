import json

import pandas as pd
from omegaconf import DictConfig

from main.structure.Filters.FilterInterface import FilterInterface
from main.tooling.FileManager import getPathForNewGeneratedFiles
from main.tooling.Logger import logging_setup
from main.tooling.MLflowHandler import uploadDatasetToMLflow

logger = logging_setup(__name__)


class PrepareDatasetForTrainingFilter(FilterInterface):
    """
        Description: This filter converts the json data into a pandas dataframe/s, that is used for training. Used in the training pipeline.
    """

    def __init__(self, conf: DictConfig):
        self.conf = conf

    def __filter__(self) -> pd.DataFrame:
        """
            Description:
                Depending on the defined different_train_test_files in the training configuration (if training and testing is processed
                for one dataset or for different datasets), this method converts the prepared json files into pandas dataframe/s.
            Args:
                None: Uses the configuration class variable
            Returns:
                pd.DataFrame or Tuple[pd.DataFrame, pd.DataFrame]: Returns either one dataframe or two dataframes in a tuple.
        """

        logger.info("-------Start Filter 'PrepareDatasetForTrainingFilter'-------")

        if not self.conf.datasources.different_train_test_files:

            singleData = []

            with open(str(getPathForNewGeneratedFiles(name=self.conf.datasources.new_datasources.training_and_testing_file_name)), 'r') as file:
                for line in file:
                    singleData.append(json.loads(line))

            singleDF = pd.DataFrame(singleData)

            uploadDatasetToMLflow(singleDF, self.conf.datasources.new_datasources.training_and_testing_file_name)

            singleDF['group'] = singleDF.index

            return singleDF

        else:

            trainData = []
            testData = []

            with open(str(getPathForNewGeneratedFiles(name=self.conf.datasources.new_datasources.training_file_name)), 'r') as file:
                for line in file:
                    trainData.append(json.loads(line))

            trainDf = pd.DataFrame(trainData)

            uploadDatasetToMLflow(trainDf, self.conf.datasources.new_datasources.training_file_name)

            trainDf['group'] = trainDf.index

            with open(str(getPathForNewGeneratedFiles(name=self.conf.datasources.new_datasources.testing_file_name)), 'r') as file:
                for line in file:
                    testData.append(json.loads(line))

            testDf = pd.DataFrame(testData)

            uploadDatasetToMLflow(testDf, self.conf.datasources.new_datasources.testing_file_name)

            testDf['group'] = testDf.index

            return trainDf, testDf
