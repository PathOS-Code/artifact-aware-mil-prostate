"""Configuration minimale du logging applicatif."""

from __future__ import annotations

import logging


def configure_logging(level: int = logging.INFO) -> None:
    """Initialise un format de logs simple pour scripts et notebooks."""

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
