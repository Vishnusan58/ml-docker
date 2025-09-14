"""
Utility functions for the ML Docker project.
"""

from pathlib import Path


def load_labels(path: str = "artifacts/labels.txt"):
    """
    Load class labels from a text file and return them as a list of strings.
    If the file does not exist, returns numeric class labels as strings.
    """
    p = Path(path)
    if p.exists():
        return p.read_text().splitlines()
    return [str(i) for i in range(10)]
