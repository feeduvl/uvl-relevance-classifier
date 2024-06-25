import pandas as pd
from omegaconf import DictConfig

from main.structure.Filters.FilterInterface import FilterInterface
from main.tooling.FileManager import getPathForNewGeneratedFiles
from main.tooling.Logger import logging_setup

logger = logging_setup(__name__)


class MergeNewGeneratedJSONFilesFilter(FilterInterface):
    """
        Description: This filter merges the data from the two new generated json files into one json file. Used in the training pipeline.
    """

    def __init__(self, conf: DictConfig):
        self.conf = conf

    def __filter__(self) -> None:
        """
           Description:
                This method reads the new generated json files as pandas dataframes, concatenates them and saves the whole data
                into a new json file.
            Args:
                None: Uses the configuration class variable
            Returns:
                None: Saves the json data as a file
        """

        logger.info("-------Start Filter 'MergeNewGeneratedJSONFilesFilter'-------")

        df1 = pd.read_json(getPathForNewGeneratedFiles(name=self.conf.datasources.new_datasources.new_excel2json_file_name), lines=True)
        df2 = pd.read_json(getPathForNewGeneratedFiles(name=self.conf.datasources.new_datasources.new_combined_original_jsons_file_name), lines=True)

        combined_df = pd.concat([df1, df2], ignore_index=True)

        combined_df.to_json(getPathForNewGeneratedFiles(name=self.conf.datasources.new_datasources.new_combined_datasets_file_name), orient='records', lines=True)
