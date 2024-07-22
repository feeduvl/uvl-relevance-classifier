from copy import deepcopy
from typing import List, Tuple

from main.structure.DataModels import Annotation, Code
from main.structure.Filters.FilterInterface import FilterInterface
from main.tooling.Logger import logging_setup

logger = logging_setup(__name__)


INFORMATIVE = "Informative"
NON_INFORMATIVE = "Non-Informative"


class ExtendAnnotationFilter(FilterInterface):
    """
        Description: This filter uses the relevance predictions for filling the empty codes list of the annotation.
        Used in the creation pipeline.
    """

    def __init__(self, annotation: Annotation):
        self.annotation = deepcopy(annotation)  # avoid changing original annotation

    def __filter__(self, sentencesAndLabels: Tuple[List[str], List[str]]) -> Annotation:
        """
            Description:
                Based on the predictions for the sentences, this method creates codes and adds them to the codes list of the annotation and
                adjusts the tokens parameters "num_name_codes" and "num_tore_codes".

            Args:
                Tuple[List[str], List[str]]: A tuple, that contains a list with the the original sentences and a list with the relevance
                predictions
            Returns:
                Annotation: The finished annotation (codes list is filled)
        """

        logger.info("-------Start Filter 'ExtendAnnotationFilter'-------")

        sentences, labels = sentencesAndLabels

        codes = self.__createCodes__(sentences, labels)

        self.__addCodes2Annotation__(codes)

        self.addCodes2Tokens(codes)

        self.annotation["tores"] = [NON_INFORMATIVE, INFORMATIVE]

        return self.annotation

    def __createCodes__(self, sentences: List[str], labels: List[str]) -> List[Code]:
        """
            Description:
                This method creates the codes list.

            Args:
                List[str]: The sentences
                List[str]: The labels
            Returns:
                List[Code]: The codes list
        """

        tokenIDX = 0
        codeIDX = 0
        codes = []
        for sentence, sentenceLabel in zip(sentences, labels, strict=True):
            code = Code(
                tokens=[tokenIDX],
                name="",
                tore=sentenceLabel,
                index=codeIDX,
                relationship_memberships=[],
            )
            codes.append(code)
            codeIDX += 1
            tokenIDX += 1

        logger.debug("-------Codes generated-------")

        return codes

    def __addCodes2Annotation__(self, codes: List[Code]) -> None:
        """
            Description:
                This method adds the codes list to the annotation.
            Args:
                List[Code]: The codes list
            Returns:
                None: The annotation class variable is adjusted
        """

        logger.debug("-------Add Codes to Annotation-------")

        self.annotation["codes"] = codes

    def addCodes2Tokens(self, codes: List[Code]) -> None:
        """
            Description:
                This method adjusts the token parameters "num_name_codes" and "num_tore_codes".
            Args:
                List[Code]: The codes list
            Returns:
                None: The annotation class variable is adjusted
        """

        logger.debug("-------Add Codes to Tokens-------")

        for code in codes:
            index = code["tokens"][0]
            self.annotation["tokens"][index]["num_name_codes"] = 0
            self.annotation["tokens"][index]["num_tore_codes"] = 1
