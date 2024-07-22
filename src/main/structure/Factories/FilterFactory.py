from typing import Any

from main.structure.Factories.FactoryInterface import FactoryInterface
from main.structure.Filters.CreateDatasetFilter import CreateDatasetFilter
from main.structure.Filters.Excel2JSONFilter import Excel2JSONFilter
from main.structure.Filters.ExtendAnnotationFilter import ExtendAnnotationFilter
from main.structure.Filters.FilterInterface import FilterInterface
from main.structure.Filters.MergeNewGeneratedJSONFilesFilter import MergeNewGeneratedJSONFilesFilter
from main.structure.Filters.MergeOriginalJSONFilesFilter import MergeOriginalJSONFilesFilter
from main.structure.Filters.PredictionFilter import PredictionFilter
from main.structure.Filters.PrepareDatasetForPredictionFilter import PrepareDatasetForPredictionFilter
from main.structure.Filters.PrepareDatasetForTrainingFilter import PrepareDatasetForTrainingFilter
from main.structure.Filters.TrainAndEvaluateModelsFilter import TrainAndEvaluateModelsFilter


class FilterFactory(FactoryInterface):
    """
        Description: Factoryclass for dynamically creating filters.
    """

    def __init__(self) -> None:
        pass

    def __create__(self, filterName: str, *args: Any, **kwargs: Any) -> FilterInterface:
        """
            Description:
                Based on the filterName, this method creates the specific filter and sets the specific parameters, which the specific filter needs.
            Args:
                str: The filter name
                Any: Not specifically defined, how many arguments are handed in, therefor the method gets access via **kwargs
            Returns:
                FilterInterface: One filter object
        """

        match filterName:
            case "Excel2JSONFilter":
                return Excel2JSONFilter(kwargs["conf"])
            case "MergeOriginalJSONFilesFilter":
                return MergeOriginalJSONFilesFilter(kwargs["conf"])
            case "MergeNewGeneratedJSONFilesFilter":
                return MergeNewGeneratedJSONFilesFilter(kwargs["conf"])
            case "PrepareDatasetForTrainingFilter":
                return PrepareDatasetForTrainingFilter(kwargs["conf"])
            case "TrainAndEvaluateModelsFilter":
                return TrainAndEvaluateModelsFilter(kwargs["conf"])
            case "PrepareDatasetForPredictionFilter":
                return PrepareDatasetForPredictionFilter(kwargs["annotation"])
            case "PredictionFilter":
                return PredictionFilter(kwargs["conf"])
            case "ExtendAnnotationFilter":
                return ExtendAnnotationFilter(kwargs["annotation"])
            case "CreateDatasetFilter":
                return CreateDatasetFilter(kwargs["content"])
            case _:
                raise ValueError(f"Filter '{filterName}' not supported")
