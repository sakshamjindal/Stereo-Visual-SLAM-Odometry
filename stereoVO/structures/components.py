"""
Author : Saksham Jindal
Date : January 5, 2020
"""

__all__ = ['StateBolts', ]

class StateBolts(object):
    
    """
    Acts as a state-storehouse which stores the left and right property of each
    """

    def __init__(self):
        self._left = None
        self._right = None
        
    @property
    def left(self):
        return self._left
    
    @left.setter
    def left(self, attribute):
        if attribute is None:
            raise ValueError("Attribute can not be None while setting the property")
        self._left = attribute

    @property
    def right(self):
        return self._right
    
    @right.setter
    def right(self, attribute):
        if attribute is None:
            raise ValueError("Attribute can not be None while setting the property")
        self._right = attribute