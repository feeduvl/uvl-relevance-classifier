from typing import List

from main.structure.DataModels import Annotation
from main.structure.Filters.FilterInterface import FilterInterface
from main.tooling.Logger import logging_setup

logger = logging_setup(__name__)


class PrepareDatasetForPredictionFilter(FilterInterface):
    """
        Description: This filter extracts annotation parts and adds these to a list of strings,that is used for the model prediction.
        Used in the creation pipeline.
    """

    def __init__(self, annotation: Annotation):
        self.annotation = annotation

    def __filter__(self) -> List[str]:
        """
            Description:
                This method extracts the names of the annotation tokens and adds them to a list.
            Args:
                None: Uses the annotation class variable
            Returns:
                List[str]: A list, that contains the token names
        """

        logger.info("-------Start Filter 'PrepareDatasetForPredictionFilter'-------")

        sentencesForPrediction = []

        for token in self.annotation['tokens']:
            sentencesForPrediction.append(token['name'])

        return sentencesForPrediction
