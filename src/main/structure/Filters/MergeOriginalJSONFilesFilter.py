import json
import os

import pandas as pd
from omegaconf import DictConfig

from main.structure.Filters.FilterInterface import FilterInterface
from main.tooling.FileManager import getPathForNewGeneratedFiles, getPathForOriginalDatasets
from main.tooling.Logger import logging_setup

logger = logging_setup(__name__)


class MergeOriginalJSONFilesFilter(FilterInterface):
    """
        Description: This filter merges important parts from the four original json files into one json file. Used in the training pipeline.
    """

    def __init__(self, conf: DictConfig):
        self.conf = conf

    def __filter__(self) -> None:
        """
            Description:
                This method loads the original json files, extracts the important parts, formates them, concatenate these formatted
                parts and save it into a new json file.
            Args:
                None: Uses the configuration class variable
            Returns:
                None: Saves the json data as a file
        """

        logger.info("-------Start Filter 'MergeOriginalJSONFilesFilter'-------")

        with getPathForOriginalDatasets(name=self.conf.datasources.original_datasources.original_relevance_app_review_json_file_name).open('r') as file:
            relevanceAppReviewJSONInputFile = json.load(file)

        with getPathForOriginalDatasets(name=self.conf.datasources.original_datasources.original_relevance_prolific1_33_json_file_name).open('r') as file:
            relevanceProlific1_33JSONInputFile = json.load(file)

        with getPathForOriginalDatasets(name=self.conf.datasources.original_datasources.original_relevance_prolific34_66_json_file_name).open('r') as file:
            relevanceProlific34_66JSONInputFile = json.load(file)

        with getPathForOriginalDatasets(name=self.conf.datasources.original_datasources.original_relevance_prolific67_100_json_file_name).open('r') as file:
            relevanceProlific67_100JSONInputFile = json.load(file)

        jsonFilesList = [relevanceAppReviewJSONInputFile, relevanceProlific1_33JSONInputFile, relevanceProlific34_66JSONInputFile, relevanceProlific67_100JSONInputFile]

        for originalJSONFile in jsonFilesList:

            tokens = originalJSONFile.get("tokens", [])
            codes = originalJSONFile.get("codes", [])
            codesFiltered = [code for code in codes if code['index']]

            for code in codesFiltered:
                for token in tokens:
                    if token["index"] == code["tokens"][0]:
                        code["tokenName"] = self.__remove_non_ascii_chars__(token["name"])
                        break

            df = pd.DataFrame(codesFiltered)

            df['labels'] = df['tore'].map({'Informative': 1, 'Non-Informative': 0})

            df = df[['tokenName', 'labels']].rename(columns={"tokenName": "text"})

            outputFilePath = getPathForNewGeneratedFiles(name=self.conf.datasources.new_datasources.new_combined_original_jsons_file_name)

            if os.path.exists(outputFilePath):
                with open(outputFilePath, 'a') as file:
                    for index, row in df.iterrows():
                        json.dump(row.to_dict(), file)
                        file.write('\n')
            else:
                with open(outputFilePath, 'w') as file:
                    for index, row in df.iterrows():
                        json.dump(row.to_dict(), file)
                        file.write('\n')

    def __remove_non_ascii_chars__(self, text: str) -> str:
        """
            Description:
                Remove the non ASCII characters and backslashes (to show ") from the text
            Args:
                str: The token text, which is a sentence
            Returns:
                str: Cleaned sentence
        """
        # ord() function takes a single character and returns an integer representing the Unicode code point of that character
        # text ASCIIs from code 32-127 / also remove backslashes
        return ''.join(char for char in text if 32 <= ord(char) < 128 and char != '\\')
