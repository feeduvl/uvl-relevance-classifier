from datetime import datetime
from typing import Dict, Tuple, cast

from main.structure.DataModels import Annotation, Code, Dataset, Documents
from main.structure.Filters.FilterInterface import FilterInterface
from main.tooling.Logger import logging_setup

logger = logging_setup(__name__)

INFORMATIVE = "Informative"


class CreateDatasetFilter(FilterInterface):
    """
        Description: This filter uses the finished annotation to create a dataset, that contains only relevant sentences.
        Used in the creation pipeline.
    """

    def __init__(self, content: Dict[str, str]):
        self.content = content

    def __filter__(self, annotation: Annotation) -> Tuple[Annotation, Dataset]:
        """
            Description:
                This method extracts relevant sentences from the annotation and creates a new dataset,
                which contains only these sentences.
            Args:
                Annotation: The finished annotation
            Returns:
                Tuple[Annotation, Dataset]: A tuple, that contains the finished annotation and the new generated dataset
        """

        logger.info("-------Start Filter 'CreateDatasetFilter'-------")

        relevantDocs = []

        for doc in annotation['docs']:
            informativeSentencesInDoc = []
            relevantDoc = {}
            for index in range(doc['begin_index'], doc['end_index']):
                codeForToken = self.__getCodeForToken__(index, annotation)
                codeRelevance = codeForToken['tore']
                if codeRelevance == INFORMATIVE:
                    sentence = self.__getTokenName__(index, annotation)
                    informativeSentencesInDoc.append(sentence)
                else:
                    pass

            if len(informativeSentencesInDoc) > 0:
                relevantDoc["id"] = doc['name']
                relevantDoc["text"] = " ".join(informativeSentencesInDoc)
                relevantDocs.append(relevantDoc)

        if len(relevantDocs) > 0:

            # now = datetime.now()
            # formatted_now = now.strftime("%Y-%m-%d_%H:%M:%S")

            newGeneratedDataset = Dataset(
                uploaded_at=datetime.now().astimezone().isoformat(),
                # name = self.content["dataset"]["name"] + "_rc_" + str(formatted_now),
                name=self.content['params']['new_dataset_name'],  # type: ignore[index]
                size=len(annotation['docs']),
                documents=cast(Documents, relevantDocs)
            )

            logger.info("-------New Dataset successfully created!-------")

            return annotation, newGeneratedDataset

        else:
            logger.info("-------New Dataset has no informative sentences and is therefor not created!-------")

            return annotation, {}

    def __getCodeForToken__(self, index: int, annotation: Annotation) -> Code:
        """
            Description:
                This method gets the tokens' code.
            Args:
                Annotation: The finished annotation
                int: The token index
            Returns:
                Code: The tokens' code
        """

        for code in annotation['codes']:
            if code['tokens'][0] == index:
                codeForTokenIndex = code
                break

        return codeForTokenIndex

    def __getTokenName__(self, index: int, annotation: Annotation) -> str:
        """
            Description:
                This method gets the tokens' name.
            Args:
                Annotation: The finished annotation
                int: The token index
            Returns:
                str: The tokens' name
        """

        for token in annotation['tokens']:
            if token['index'] == index:
                tokenName = token['name']
                break

        return tokenName
