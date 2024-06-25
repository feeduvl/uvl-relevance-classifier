from typing import Any

from main.structure.Factories.FactoryInterface import FactoryInterface
from main.structure.Pipelines.CreationPipeline import CreationPipeline
from main.structure.Pipelines.Pipeline import Pipeline
from main.structure.Pipelines.TrainingPipeline import TrainingPipeline


class PipelineFactory(FactoryInterface):
    """
        Description: Factoryclass for dynamically creating pipelines.
    """

    def __init__(self) -> None:
        pass

    def __create__(self, pipelineName: str, *args: Any, **kwargs: Any) -> Pipeline:
        """
            Description:
                Based on the pipelineName, this method creates the specific pipeline and sets the specific parameters,
                which the specific pipeline needs.
            Args:
                str: The pipeline name.
                Any: Not specifically defined, how many arguments are handed in, therefor the method gets access via **kwargs.
            Returns:
                Pipeline: One pipeline object.
        """

        match pipelineName:
            case "TrainingPipeline":
                return TrainingPipeline(kwargs["conf"])
            case "CreationPipeline":
                return CreationPipeline(kwargs["conf"], kwargs["content"], kwargs["annotation"])
            case _:
                raise ValueError(f"Retriever {pipelineName} not supported")
