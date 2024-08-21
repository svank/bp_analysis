from enum import IntEnum, auto


class Flag(IntEnum):
    GOOD = 1
    FALSE_POS = 2
    CLOSE_NEIGHBOR = 3
    EDGE = 4
    TOO_SMALL = 5
    TOO_BIG = 6
    TOO_LONG = 7
    
    def __repr__(self):
        return self.name
    
    def __str__(self):
        str = self.name.lower()
        str = str.replace('_', ' ')
        str = str[0].upper() + str[1:]
        return str


class SequenceFlag(IntEnum):
    GOOD = 100
    TOO_SHORT = 101
    
    def __repr__(self):
        return self.name
    
    def __str__(self):
        str = self.name.lower()
        str = str.replace('_', ' ')
        str = str[0].upper() + str[1:]
        return str


class Event(IntEnum):
    NORMAL = 200
    MERGE = 201
    SPLIT = 202
    FIRST_IMAGE = 203
    LAST_IMAGE = 204
    COMPLEX = 205
    SIZE_CHANGE_PX = 206
    SIZE_CHANGE_PCT = 207
    
    def __repr__(self):
        return self.name
    
    def __str__(self):
        str = self.name.lower()
        str = str.replace('_', ' ')
        str = str[0].upper() + str[1:]
        return str
