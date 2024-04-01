import sys
import logging

import time

def make_logger(name, filename=None, level=logging.INFO):
    logger = logging.getLogger(name)
    if filename:
        logging.basicConfig(filename=filename, level=level)
    else:
        logging.basicConfig(level=level)
    format = logging.Formatter("%(asctime)s | %(message)s")
    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(format)
    logger.addHandler(sh)
    return logger


def time_to_string(seconds):
    return time.strftime("%H:%M:%S", time.localtime(seconds + 57600))
