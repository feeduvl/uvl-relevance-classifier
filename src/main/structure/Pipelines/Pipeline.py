from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List

from omegaconf import DictConfig

from main.structure.Filters.FilterInterface import FilterInterface
from main.tooling.FileManager import cleanup
from main.tooling.Logger import logging_setup

logger = logging_setup(__name__)


class Pipeline(ABC):
    """
        Description: Interface for pipeline classes. Defines, that every pipeline class has to implement the "__compose__" method.
        All Pipeline objects implement the same "__process__" method, which is here defined.
    """

    @abstractmethod
    def __init__(self, conf: DictConfig) -> None:
        self.conf = conf
        self.pipelineFilters: List[FilterInterface] = []

    @abstractmethod
    def __compose__(self) -> None:
        raise NotImplementedError("Subclasses must implement this method!")

    def __process__(self) -> Any:
        """
            Description:
                This method processes the pipeline filters. This is done via a loop, that hands in the return object of the previous filter
                to the next filter. At the end, the new generated files are deleted.
            Args:
                None: The class variable is used
            Returns:
                Any: Depends on the pipeline, what is returned

        """

        logger.info("-------Pipeline started-------")

        filterResult = None
        for filter in self.pipelineFilters:
            if filterResult is not None:
                # Call the function with the return values as arguments
                filterResult = filter.__filter__(filterResult)
            else:
                # Call the function without any arguments
                filterResult = filter.__filter__()

        if (self.conf.training):
            cleanup()

        logger.info("-------Pipeline finished-------")

        return filterResult
