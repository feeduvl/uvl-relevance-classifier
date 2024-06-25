from typing import List

from omegaconf import DictConfig

from main.structure.Factories.FilterFactory import FilterFactory
from main.structure.Filters.FilterInterface import FilterInterface
from main.structure.Pipelines.Pipeline import Pipeline
from main.tooling.Logger import logging_setup

logger = logging_setup(__name__)


class TrainingPipeline(Pipeline):
    """
        Description: This pipeline is specifically for training the relevance classification model.
    """

    def __init__(self, conf: DictConfig):
        self.pipelineFilters: List[FilterInterface] = []
        self.__compose__(conf)

    def __compose__(self, conf: DictConfig) -> None:
        """
            Description:
                From the list, that contains the filter names (in the conf), this method uses the FilterFactory, to create a new
                list, which contains the filter objects and sets it as a class variable, which is used in the "__process__" method.
            Args:
                DictConfig: The configuration with the filter name list in it
            Returns:
                None: Fills the class variable pipelineFilters with real filter objects
        """

        filterFactory = FilterFactory()
        for filterName in conf.filterList:
            self.pipelineFilters.append(filterFactory.__create__(filterName, conf=conf))
