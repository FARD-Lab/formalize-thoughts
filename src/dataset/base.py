from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator

class BaseLoader(ABC):
    """Abstract base class for dataset loaders."""

    @abstractmethod
    def load(self) -> None:
        """Load dataset from source.

        This method should handle all dataset loading logic including
        downloading, caching, and initial setup.
        """
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over dataset examples.

        Yields:
            Dictionary with example data
        """
        pass

    @abstractmethod
    def get_text_field(self, example: Dict[str, Any]) -> str:
        """Extract the main text field from a dataset example.

        Args:
            example: A dictionary representing a single dataset example.
        Returns:
            The text content of the example.
        """
        pass

    @abstractmethod
    def get_metadata(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from a dataset example.

        Args:
            example: A dictionary representing a single dataset example.
        Returns:
            A dictionary containing metadata of the example.
        """
        pass

    @abstractmethod
    def reject_example(self, example: Dict[str, Any]):
        """Handle rejection of an example.

        This method can be used to update internal counters or states
        when an example is rejected during processing.

        Args:
            example: A dictionary representing a single dataset example.
        """
        pass

    @property
    @abstractmethod
    def instruction(self) -> str:
        """Return the instruction associated with the dataset, if any."""
        pass
