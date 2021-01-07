"""
Author : Saksham Jindal
Date : January 5, 2020
"""

__all__ = ['StateBolts', ]

class StateBolts(object):

    """
    Implementation of state-storehouse (data structure) which stores the left and right property 
    of properties of the VO State (VO_StateMachine) at one time instance
    """

    def __init__(self, _left = None, _right = None):
        self._left = _left
        self._right = _right
        
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