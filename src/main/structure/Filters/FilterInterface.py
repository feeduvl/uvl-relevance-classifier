from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class FilterInterface(ABC):
    """
        Description: Interface for filter classes. Defines, that every filter class has to implement the "__filter__" method.
    """

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __filter__(self) -> Any:
        pass
