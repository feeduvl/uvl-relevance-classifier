from copy import deepcopy
from typing import List, Tuple
from datetime import datetime

import nltk
from nltk.tokenize import word_tokenize

from main.structure.DataModels import Annotation

INFORMATIVE = "Informative"
NON_INFORMATIVE = "Non-Informative"
NO_CODE = "0"

# Download the NLTK tokenizer data
nltk.download('punkt')


class AnnotationMapper():

    def __init__(self, wordBasedAnnotation: Annotation, sentenceBasedAnnotation: Annotation) -> None:
        self.wordBasedAnnotation = deepcopy(wordBasedAnnotation)  # avoid changing original wordBasedAnnotation
        self.sentenceBasedAnnotation = sentenceBasedAnnotation

    def mapRelevantSentences2WordBasedAnnotation(self) -> Annotation:
        """
            Description:
                The behavior function to adjust the wordBasedAnnotation, that it only contains informative sentences.
            Args:
                Annotation: The original class variable wordBasedAnnotation
            Returns:
                Annotation: The updated class variable wordBasedAnnotation
        """

        sentencesAndLabels = self.extractSentencesAndLabels()

        wordsAndLabels = self.wordTokenizeSentencesAndLabelWords(sentencesAndLabels)

        self.extendTokensParameters(wordsAndLabels)

        tokensToRemoveIndices = self.extractTokensToRemove()

        self.adjustDocs()

        # Delete unrelevant tokens from self.wordBasedAnnotation["tokens"] based on tokensToRemoveIndices
        self.wordBasedAnnotation["tokens"] = [token for token in self.wordBasedAnnotation["tokens"] if token["index"] not in tokensToRemoveIndices]

        self.adjustCodes(tokensToRemoveIndices)

        self.removeRelevanceParameterFromTokens()

        self.refreshTokenIndices()

        self.wordBasedAnnotation["name"] = self.wordBasedAnnotation["name"] + "_Relevance_Mapping"
        
        now = datetime.now()
        
        formattedNow = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        self.wordBasedAnnotation["uploaded_at"] = formattedNow

        return self.wordBasedAnnotation

    def extractSentencesAndLabels(self) -> List[Tuple]:
        """
            Description:
                Extract the information from the sentenceBasedAnnotation, which sentence is informative or non-informative
                and put this into a data structure.
            Args:
                Annotation: self.sentenceBasedAnnotation
            Returns:
                List[Tuple]: List with Tuples, that looks like "[(<Sentence1>, <Relevance>), (<Sentence2>, <Relevance>), ...]"
        """

        tokens = self.sentenceBasedAnnotation['tokens']
        codes = {code['tokens'][0]: code['tore'] for code in self.sentenceBasedAnnotation['codes'] if code['index']}
        return [(token['name'], codes.get(token['index'], '0')) for token in tokens]

    def wordTokenizeSentencesAndLabelWords(self, sentencesAndLabels: List[Tuple]) -> List[Tuple]:
        """
            Description:
                Word-tokenize each sentence from the sentencesAndLabels tuple list, assign the same label to a
                word as the sentence in which it is contained and put this into a data structure
                sentence.
            Args:
                List[Tuple]: sentencesAndLabels (the output from extractSentencesAndLabels())
            Returns:
                List[Tuple]: List with Tuples, that looks like
                "[(Word1FromSentence1>, <RelevanceFromSentence1>), (<Word2FromSentence1>, <RelevanceFromSentence1>), ..., (<Word1FromSentence2>, <RelevanceFromSentence2>), (<Word2FromSentence2>, <RelevanceFromSentence2>), ...]"
        """

        wordsAndLabels = []
        for sentence, label in sentencesAndLabels:
            words = word_tokenize(sentence)
            wordsAndLabels.extend([(word, label) for word in words])
        return wordsAndLabels

    def extendTokensParameters(self, wordsAndLabels: List[Tuple]) -> None:
        """
            Description:
                Iterate through the token list of wordBasedAnnotation['tokens'], add the parameter "relevance" to every
                token and assign the same value, which is contained for that token in wordsAndLabels.

                For example a token looks now like:
                {'index': 6, 'name': 'app', 'lemma': 'app', 'pos': 'n', 'num_name_codes': 1, 'num_tore_codes': 1, 'relevance': 'Informative'}
            Args:
                Annotation: self.wordBasedAnnotation
                List[Tuple]: wordsAndLabels (the output from wordTokenizeSentencesAndLabelWords())
            Returns:
                None: The class variable wordBasedAnnotation is updated
        """

        for token, (_, label) in zip(self.wordBasedAnnotation['tokens'], wordsAndLabels):
            self.wordBasedAnnotation['tokens'][token["index"]]["relevance"] = label

    def extractTokensToRemove(self) -> List[int]:
        """
            Description:
                Extract token indices, which have the entry "Non-Informative" for the "relevance" parameter in the
                wordBasedAnnotation
            Args:
                Annotation: self.wordBasedAnnotation
            Returns:
                List[int]: Token indices, that are "Non-Informative"
        """

        tokensToRemoveIndices = []

        for token in self.wordBasedAnnotation["tokens"]:
            if token["relevance"] == NON_INFORMATIVE or token["relevance"] == NO_CODE:
                tokensToRemoveIndices.append(token["index"])

        return tokensToRemoveIndices

    def adjustDocs(self) -> None:
        """
            Description:
                Adjust the docs "begin_index" and "end_index" based of the number of unrelevant words in this doc,
                delete a doc if it contains only unrelevant words, update the annotation "size" parameter
            Args:
                Annotation: self.wordBasedAnnotation
            Returns:
                None: The class variable wordBasedAnnotation is updated
        """

        # for example: {0: 3, 1: 26, 2: 113} -> means doc0 has 3 unrelevant words, doc1 has 26 unrelevant words and
        # doc2 has 113 unrelevant words
        doc2NumberOfUnrelevantWordsDict = {}

        for docIdx, doc in enumerate(self.wordBasedAnnotation["docs"]):
            unrelevantWordsCounter = 0
            for token in self.wordBasedAnnotation["tokens"][doc["begin_index"]:doc["end_index"]]:
                if token["relevance"] == NON_INFORMATIVE or token["relevance"] == NO_CODE:
                    unrelevantWordsCounter += 1
            doc2NumberOfUnrelevantWordsDict[docIdx] = unrelevantWordsCounter

        for docIdxNow, doc in enumerate(self.wordBasedAnnotation["docs"]):

            doc["end_index"] -= doc2NumberOfUnrelevantWordsDict[docIdxNow]

            for docIdxNext in range(docIdxNow + 1, len(self.wordBasedAnnotation["docs"])):
                nextDoc = self.wordBasedAnnotation["docs"][docIdxNext]
                nextDoc["begin_index"] -= doc2NumberOfUnrelevantWordsDict[docIdxNow]
                nextDoc["end_index"] -= doc2NumberOfUnrelevantWordsDict[docIdxNow]

        # If a doc has the same value for "begin_index" and "end_index", it means it contains no relevant words and
        # is therefor not relevant and is deleted
        docsToKeep = []
        for doc in self.wordBasedAnnotation["docs"]:
            if doc["begin_index"] != doc["end_index"]:
                docsToKeep.append(doc)

        self.wordBasedAnnotation["docs"] = docsToKeep

        self.wordBasedAnnotation["size"] = len(docsToKeep)

    def adjustCodes(self, tokensToRemoveIndices: List[int]) -> None:
        """
            Description:
                Extract all the codes, that should be kept. In this case, a code contains not only unrelevant
                words (the indices in a code["tokens"] list, contains not only indices, that are included in
                the tokensToRemoveIndices list)
            Args:
                Annotation: self.wordBasedAnnotation
                List[int]: tokensToRemoveIndices (the output from extractTokensToRemove())
            Returns:
                None: The class variable wordBasedAnnotation is updated
        """

        codesToKeep = []
        for code in self.wordBasedAnnotation["codes"]:
            code["tokens"] = [token for token in code["tokens"] if token not in tokensToRemoveIndices]

            if code["tokens"]:
                codesToKeep.append(code)

        self.wordBasedAnnotation["codes"] = codesToKeep

        new_code_index = 0
        for code in self.wordBasedAnnotation["codes"]:
            code["index"] = new_code_index
            new_code_index += 1

    def removeRelevanceParameterFromTokens(self) -> None:
        """
            Description:
                Remove the relevance parameter from all tokens, that was added in extendTokensParameters()
            Args:
                Annotation: self.wordBasedAnnotation
            Returns:
                None: The class variable wordBasedAnnotation is updated
        """

        for token in self.wordBasedAnnotation["tokens"]:
            del token["relevance"]

    def refreshTokenIndices(self) -> None:
        """
            Description:
                The token indices must be adjusted, so that the indices are ascending in the token list of the annotation.
                To do this, first the dict oldToNewTokenIndices is created, that contains the mapping from old token index
                to new token index. Second, the token index references in the code["tokens"] lists are adjusted according to
                the mapping in oldToNewTokenIndices. Finally, the token indices are adjusted according to the mapping oldToNewTokenIndices.
            Args:
                Annotation: self.wordBasedAnnotation
            Returns:
                None: The class variable wordBasedAnnotation is updated
        """

        oldToNewTokenIndices = {}
        newTokenIndex = 0
        for token in self.wordBasedAnnotation["tokens"]:
            oldToNewTokenIndices[token["index"]] = newTokenIndex
            newTokenIndex += 1

        for code in self.wordBasedAnnotation["codes"]:
            newTokens = []
            for oldTokenIndex in code["tokens"]:
                if oldTokenIndex in oldToNewTokenIndices:
                    newTokens.append(oldToNewTokenIndices[oldTokenIndex])
            code["tokens"] = newTokens

        for token in self.wordBasedAnnotation["tokens"]:
            token["index"] = oldToNewTokenIndices[token["index"]]
