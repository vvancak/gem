TEMPLATE = '\033[{}m{}\033[0m'
ERROR = TEMPLATE.format(31, "[ERROR]:")
WARN = TEMPLATE.format(33, "[WARN]:")
INFO = TEMPLATE.format(0, "[INFO]:")
OK = TEMPLATE.format(32, "[OK]:")
