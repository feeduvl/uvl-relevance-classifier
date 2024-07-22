from omegaconf import DictConfig, OmegaConf

from main.structure.Factories.FactoryInterface import FactoryInterface
from main.tooling.FileManager import getConfigPath


class ConfigurationFactory(FactoryInterface):
    """
        Description: Factoryclass for dynamically creating configurations.
    """

    def __init__(self) -> None:
        pass

    def __create__(self, configName: str) -> DictConfig:
        """
            Description:
                Based on the configName, this method creates the specific configuration.
            Args:
                str: The configuration name
            Returns:
                DictConfig: The configuration object from OmegaConf
        """

        match configName:
            case "TrainingAllDatasets":
                return OmegaConf.load(getConfigPath("training_all_datasets_config.yaml"))
            case "TrainingOnlyP2GoldenDataset":
                return OmegaConf.load(getConfigPath("training_only_p2golden_config.yaml"))
            case "TrainingOnlyKomootDataset":
                return OmegaConf.load(getConfigPath("training_only_komoot_dataset_config.yaml"))
            case "TrainingKomootTestP2GoldenDatasets":
                return OmegaConf.load(getConfigPath("trainingKomoot_testP2Golden_config.yaml"))
            case "TrainingP2GoldenTestKomootDatasets":
                return OmegaConf.load(getConfigPath("trainingP2Golden_testKomoot_config.yaml"))
            case "OnlyAnnotation":
                return OmegaConf.load(getConfigPath("creation_annotation_config.yaml"))
            case "OnlyDataset":
                return OmegaConf.load(getConfigPath("creation_dataset_config.yaml"))
            case "AnnotationAndDataset":
                return OmegaConf.load(getConfigPath("creation_annotation_and_dataset_config.yaml"))
            case _:
                raise ValueError(f"Filter '{configName}' not supported")
