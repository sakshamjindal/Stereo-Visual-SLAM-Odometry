from .components import StateBolts

__all__ = ['VO_StateMachine']


class VO_StateMachine():

    """
    The VO_state acts as a state-machine which stores the
    frames, descriptors, tracked features and matched features
    """

    def __init__(self, state_num=None, params=None):
    
        """
        Args:
            state_num : the number of the frame
        
        """
        self.state_num = state_num
        self.params = params

        # manage information about frames 
        self.frames = StateBolts()
        
        # maange the results of matched descriptors
        self.matchedPoints = StateBolts()
        self.keypoints = StateBolts()
        self.descriptors = StateBolts()
        self.inliers = StateBolts()
        
        # manage triangulation results
        self.pts3D = None
        self.pts3D_Filter = None
        self.InliersFilter = StateBolts()
        
        # manage tracking results
        self.trackedPoints_further = StateBolts()
        self.trackerPoints_P3P = StateBolts()
        
        # manage PnP (or P3P) filtered results
        self.InliersP3P = StateBolts()
        self.P3P_pts3D = None
        
        # manage 6DOF pose in the current state
        # Location_TransVec = None
        # Orientation_RotMat = None

    def __setattr__(self, name, value):
        if name in self.__dict__:
            if isinstance(self.__dict__[name], StateBolts) and isinstance(value, tuple):
                left_property = value[0]
                right_propery = value[1]
                self.__dict__[name].left = left_property
                self.__dict__[name].right = right_propery
            else:
                self.__dict__[name] = value
        else:
            self.__dict__[name] = value
    
    def none_checks(self):
        """
        Call the function before moving on to the next state and assigning (prevState<-currState)
        """
        if True:
            raise ValueError(" ... ")
            