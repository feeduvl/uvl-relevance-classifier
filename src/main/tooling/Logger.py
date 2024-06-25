import logging


def logging_setup(name: str) -> logging.Logger:

    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)s:%(message)s",  # Output Format
        level=logging.INFO,  # From the INFO priority onwards, everything should be logged
        datefmt="%I:%M:%S",  # Date/Time-Format - here only time
    )

    logger = logging.getLogger(name)

    return logger
