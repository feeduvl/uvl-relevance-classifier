from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class FactoryInterface(ABC):
    """

    """

    @abstractmethod
    def __create__(self, creationName: str, *args: Any, **kwargs: Any) -> Any:
        """
        Description: Interface for factory classes. Defines, that every factory class has to implement the "__create__" method.
        """
        pass
