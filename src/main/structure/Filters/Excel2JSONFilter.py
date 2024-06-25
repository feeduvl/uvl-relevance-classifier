import pandas as pd
from omegaconf import DictConfig

from main.structure.Filters.FilterInterface import FilterInterface
from main.tooling.FileManager import getPathForNewGeneratedFiles, getPathForOriginalDatasets
from main.tooling.Logger import logging_setup

logger = logging_setup(__name__)


class Excel2JSONFilter(FilterInterface):
    """
        Description: This filter converts the original excel data into json data. Used in the training pipeline.
    """

    def __init__(self, conf: DictConfig):
        self.conf = conf

    def __filter__(self) -> None:
        """
            Description:
                This method reads the original excel data as a pandas dataframe and converts it into json data. The json
                data is saved in a .json file.
            Args:
                None: Uses the configuration class variable
            Returns:
                None: Saves the json data as a file
        """

        logger.info("-------Start Filter 'Excel2JSONFilter'-------")

        df = pd.read_excel(getPathForOriginalDatasets(name=self.conf.datasources.original_datasources.original_excel_file_name), header=None, names=['text', 'labels'], skiprows=1)
        df.to_json(getPathForNewGeneratedFiles(name=self.conf.datasources.new_datasources.new_excel2json_file_name), orient='records', lines=True)
