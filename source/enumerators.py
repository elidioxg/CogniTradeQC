#region imports
from AlgorithmImports import *
from enum import Enum
#endregion

SIGNAL_NONE = 1
SIGNAL_LONG = 2
SIGNAL_SHORT = 3
SIGNAL_LONG_STRONG = 4
SIGNAL_SHORT_STRONG = 5

#region enums
class SignalOnSameDirection(Enum):

    NONE = 0
    EMMIT_SIGNAL = 5
    EMMIT_IF_NOT_PROFITABLE = 6
    EMMIT_IF_STRONG_SIGNAL = 7
    EMMIT_IF_STRONG_SIGNAL_AND_NOT_PROFITABLE = 8

class SignalOnOppositeDirection(Enum):

    NONE = 0
    CLOSE = 3
    CLOSE_IF_PROFITABLE = 4
    EMMIT_SIGNAL = 5
    EMMIT_IF_STRONG_SIGNAL = 7
    #EMMIT_IF_STRONG_SIGNAL_AND_PROFITABLE = 9

class SignalNeutral(Enum):

    NONE = 0
    CLOSE = 3
    CLOSE_IF_PROFITABLE = 4
    EMMIT_SIGNAL = 5

class TrainingFrequency(Enum):

    DAILY = 0
    WEEKLY = 1
    MONTHLY = 2

class SignalCheckFrequency(Enum):

    AFTER_MARKET_OPEN = 1
    BEFORE_MARKET_CLOSE = 2
    TWICE_A_DAY = 3

class PortfolioConstructor(Enum):

    ACCUMULATIVE_INSIGHT = 1
    CONFIDENCE_WEIGHTED = 2
    EQUAL_WEIGHTING = 3
    INSIGHT_WEIGHTING = 4

class OrderExecutionType(Enum):

    IMMEDIATE = 1
    VOLUME_WEIGHTED_AVERAGE_PRICE = 2

#endregion
