from enum import Enum


class BatchType(Enum):
    HEAD_BATCH = 0
    TAIL_BATCH = 1
    SINGLE = 2