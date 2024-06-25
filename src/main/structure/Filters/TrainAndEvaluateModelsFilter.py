from typing import Any, Dict, List, Tuple, Union

import pandas as pd
from datasets import Dataset, DatasetDict
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from transformers import BertForSequenceClassification, BertTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments
from transformers.trainer_utils import EvalPrediction

from main.structure.Filters.FilterInterface import FilterInterface
from main.tooling.FileManager import getModelPath, getPathForNewGeneratedFiles
from main.tooling.Logger import logging_setup
from main.tooling.MLflowHandler import computeExperimentMetricsForMLFlowUpload, createConfusionMatrixPngForMLFlowUpload, createTrainTestFileForMLFlowUpload, log_mlflow_artifacts

logger = logging_setup(__name__)


class TrainAndEvaluateModelsFilter(FilterInterface):
    """
        Description: This filter trains the model and evaluates the results. Used in the training pipeline.
    """

    def __init__(self, conf: DictConfig):
        self.conf = conf

    def __filter__(self, df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]) -> None:
        """
            Description:
                This method is the main part for training the models. The training is is processed with cross validation.
                Before training the model, the data is tokenized with [CLS] and [SEP] token added, padding, truncation and
                attention masking. For training, the Trainer from transformers library is used.
            Args:
                None: Uses the configuration class variable
            Returns:
                None: The model is saved and the evaluation results are handed in to the MLflowHandler for upload
        """

        logger.info("-------Start Filter 'TrainAndEvaluateModelsFilter'-------")

        if isinstance(df, tuple):
            initialTrainDF, initialTestDF = df
        else:
            initialTrainDF = df

        if self.conf.datasources.different_train_test_files:

            logger.info("-------Different Test Dataset!-------")

            gkfTestDataset = GroupKFold(n_splits=self.conf.fold_number)

            testIdxList = []

            for foldNumber, (_, test_idx) in enumerate(gkfTestDataset.split(initialTestDF, initialTestDF['labels'], initialTestDF['group']), start=1):
                testIdxList.append(test_idx)

        gkfTrainTestDataset = GroupKFold(n_splits=self.conf.fold_number)

        metricsList = []

        # Splitting the data
        for foldNumber, (train_idx, test_idx) in enumerate(gkfTrainTestDataset.split(initialTrainDF, initialTrainDF['labels'], initialTrainDF['group']), start=1):
            logger.info(f"-------Starting Iteration: {foldNumber}-------")

            trainDF = initialTrainDF.iloc[train_idx]

            if not self.conf.datasources.different_train_test_files:
                testDF = initialTrainDF.iloc[test_idx]

            else:
                testIDX = testIdxList[foldNumber - 1]
                testDF = initialTestDF.iloc[testIDX]

            trainDataset = Dataset.from_pandas(trainDF)
            testDataset = Dataset.from_pandas(testDF)

            datasetTrainTest = DatasetDict({'train': trainDataset, 'test': testDataset})
            datasetTrainTest['train'] = datasetTrainTest['train'].remove_columns([col for col in datasetTrainTest['train'].column_names if col not in ['text', 'labels']])
            datasetTrainTest['test'] = datasetTrainTest['test'].remove_columns([col for col in datasetTrainTest['test'].column_names if col not in ['text', 'labels']])

            createTrainTestFileForMLFlowUpload(foldNumber, datasetTrainTest)

            tokenizer = BertTokenizer.from_pretrained(self.conf.tokenizer.name)

            def preprocess_function(data: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
                return tokenizer(data["text"], truncation=True, padding=True)

            tokenized_dataset = datasetTrainTest.map(preprocess_function, batched=True)

            tokenized_dataset = tokenized_dataset.remove_columns(["text", "token_type_ids"])

            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

            def compute_metrics(eval_preds: EvalPrediction) -> Dict[str, Any]:
                labels = eval_preds.label_ids
                preds = eval_preds.predictions.argmax(-1)

                accuracy = accuracy_score(labels, preds)
                precision_label_0 = precision_score(labels, preds, pos_label=0, average='binary')
                recall_label_0 = recall_score(labels, preds, pos_label=0, average='binary')
                f1_label_0 = f1_score(labels, preds, pos_label=0, average='binary')
                precision_label_1 = precision_score(labels, preds, pos_label=1, average='binary')
                recall_label_1 = recall_score(labels, preds, pos_label=1, average='binary')
                f1_label_1 = f1_score(labels, preds, pos_label=1, average='binary')
                cm = confusion_matrix(labels, preds)

                return {
                    'accuracy': accuracy,
                    'precision_label_0': precision_label_0,
                    'recall_label_0': recall_label_0,
                    'f1_label_0': f1_label_0,
                    'precision_label_1': precision_label_1,
                    'recall_label_1': recall_label_1,
                    'f1_label_1': f1_label_1,
                    'confusion_matrix': cm.tolist()
                }

            # ###################################
            # ###########TRAINING################
            # ###################################

            logger.info("-------Training started-------")

            id2label = {0: "Non-Informative", 1: "Informative"}
            label2id = {"Non-Informative": 0, "Informative": 1}

            # init the model
            model = BertForSequenceClassification.from_pretrained(self.conf.modelArgs.name, num_labels=2, id2label=id2label, label2id=label2id)

            arguments = TrainingArguments(
                output_dir=getPathForNewGeneratedFiles("checkpoints"),  # where to save the logs and checkpoints
                per_device_train_batch_size=self.conf.modelArgs.per_device_train_batch_size,  # batch size per GPU or CPU
                per_device_eval_batch_size=self.conf.modelArgs.per_device_eval_batch_size,
                num_train_epochs=self.conf.modelArgs.num_train_epochs,  # 15 bis 25
                eval_strategy="epoch",  # steps" (evaluate every eval_steps) or "epoch" (evaluate at the end of each epoch)
                save_strategy="epoch",  # save the model at the end of each epoch
                learning_rate=self.conf.modelArgs.learning_rate,  # 2e-5 oder 2e-4
                weight_decay=self.conf.modelArgs.weight_decay,
                optim="adamw_torch",
                load_best_model_at_end=True,  # the best model based on the metric
                # seed=224,
                logging_steps=self.conf.modelArgs.logging_steps,
                push_to_hub=False
            )

            trainer = Trainer(
                model=model,
                args=arguments,
                train_dataset=tokenized_dataset['train'],
                eval_dataset=tokenized_dataset['test'],
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics
            )

            trainer.train()

            modelName = f"Iteration_{str(foldNumber)}_{self.conf.project.model_name}"

            trainer.save_model(
                output_dir=getModelPath(name=modelName)
            )

            log_mlflow_artifacts(getModelPath(name=modelName))

            # ##################################
            # ##########EVALUATION##############
            # ##################################

            logger.info("-------Evaluation started-------")

            evaluationResults = trainer.evaluate()
            logger.info(f"Evaluation results for fold {foldNumber}: {evaluationResults}")
            metricsList.append(evaluationResults)

            createConfusionMatrixPngForMLFlowUpload(foldNumber, evaluationResults, False)

            predictionsResults = trainer.predict(tokenized_dataset['test'])
            logger.info(f"Prediction results for fold {foldNumber} and the test dataset: {predictionsResults}")

            logger.info(f"-------Finished Iteration: {foldNumber}-------")

        computeExperimentMetricsForMLFlowUpload(metricsList)
