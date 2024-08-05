from enum import Enum, auto


class Flag(Enum):
    GOOD = auto()
    FALSE_POS = auto()
    CLOSE_NEIGHBOR = auto()
    EDGE = auto()
    TOO_SMALL = auto()
    TOO_BIG = auto()
    
    def __repr__(self):
        return self.name
    
    def __str__(self):
        return repr(self)


class SequenceFlag(Enum):
    GOOD = auto()
    TOO_SHORT = auto()
    
    def __repr__(self):
        return self.name
    
    def __str__(self):
        return repr(self)


class Event(Enum):
    NORMAL = auto()
    MERGE = auto()
    SPLIT = auto()
    FIRST_IMAGE = auto()
    LAST_IMAGE = auto()
    COMPLEX = auto()
    SIZE_CHANGE_PX = auto()
    SIZE_CHANGE_PCT = auto()
    
    def __repr__(self):
        return self.name
    
    def __str__(self):
        return repr(self)
