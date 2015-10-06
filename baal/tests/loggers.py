import logging

members = ["treecut", "hlfdebug"]
levels = level = {"debug": logging.DEBUG, "warning":logging.WARNING,
                  "info": logging.INFO, "error":logging.ERROR,
                  "critical":logging.CRITICAL}

def shell_logs(loggername="", level="debug"):
    logger = logging.getLogger(loggername)
    ch = logging.StreamHandler()
    ch.setLevel(levels[level])
    logger.addHandler(ch)
    logger.setLevel(levels[level])

