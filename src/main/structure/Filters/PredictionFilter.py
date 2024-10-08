from typing import List, Tuple

import torch
from omegaconf import DictConfig
from transformers import BertForSequenceClassification, BertTokenizer

from main.structure.Filters.FilterInterface import FilterInterface
from main.tooling.FileManager import getModelPath
from main.tooling.Logger import logging_setup

logger = logging_setup(__name__)

BATCH_SIZE = 100

class PredictionFilter(FilterInterface):
    """
        Description: This filter uses the trained model and categorizes sentences for their relevance.
        Used in the creation pipeline.
    """

    def __init__(self, conf: DictConfig):
        self.conf = conf

    def __filter__(self, sentences: List[str]) -> Tuple[List[str], List[str]]:
        """
            Description:
                This method tokenizes the sentences and gives them as input to the fine-tuned model, which outputs the predictions
                regarding the relevance of each sentence.
            Args:
                List[str]: A list, that contains the sentences
            Returns:
                Tuple[List[str], List[str]]: A tuple, that contains a list with the original sentences and a list with the relevance
                predictions
        """

        logger.info("-------Start Filter 'PredictionFilter'-------")

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        finetunedModel = BertForSequenceClassification.from_pretrained(getModelPath(self.conf.model_name))

        predictions = []
        for i in range(0, len(sentences), BATCH_SIZE):
            batchSentences = sentences[i:i + BATCH_SIZE]
            batchModelInputs = tokenizer(batchSentences, return_tensors="pt", padding=True, truncation=True)

            with torch.no_grad():
                batchPredictions = torch.argmax(finetunedModel(**batchModelInputs).logits, dim=1)
            
            predictions.extend(batchPredictions.detach())
            del batchModelInputs, batchPredictions


        predictedLabels = []

        for prediction in predictions:
            predictedLabels.append(["Non-Informative", "Informative"][prediction.item()])

        return sentences, predictedLabels
