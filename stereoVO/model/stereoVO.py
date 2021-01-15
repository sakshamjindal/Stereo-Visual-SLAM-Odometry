from stereoVO.structures import VO_StateMachine
from stereoVO.model.drivers import StereoDrivers


class StereoVO(StereoDrivers):

    """
    StereoVo : is the main driver code which calls upon drivers codes for DetectionEngine, TrackingEngine, PnP solver, Optimization
    to calculate the relative location and orientation of a stereo state at particular time instant
    """

    def __init__(self, intrinsic, PL, PR, params):

        """
        :param intrinsic (np.array): size (3x3) camera calibration parameters
        :param PL (np.array): size (3x4) left projection matrix such that x_L = PL * X_w
        :param PR (np.array): size (3x4) right projection matrix such that x_R = PR * X_w
                              (where world coordinates are in the frame of the left camera)
        :param params (AttriDict): contains parameters for the stereo configuration, 
                                   detection of features, tracking and other geomtric
                                   computer vision features
        """

        self.intrinsic = intrinsic
        self.PL = PL
        self.PR = PR
        self.params = params

    def __call__(self, left_frame, right_frame, state_num):

        """
        Used for calling the object of stereVO and processing the stereo states

        :param left_frame (np.array): of size (HxWx3) of stereo configuration
        :param right_frame (np.array): of size (HxWx3) of stereo configuation
        :param state_num (int): counter for specifying the frame number of stereo state

        Returns
            VO_StateMachine::location (np.array): size (3) - translation of stereo state (left camera) relative to initial location
            VO_StateMachine::orientation (np.array): size (3x3) - rotation of stereo state (right camera) relative to intial orientation
        """

        if state_num == 0:
            self._process_first_frame(left_frame, right_frame, state_num)
            return self.prevState.location, self.prevState.orientation
        if state_num == 1:
            self._process_second_frame(left_frame, right_frame, state_num)
        else:
            self._process_continuous_frame(left_frame, right_frame, state_num)

        # To Do : Add logger object instead
        print("Frame {} Processing Done ...................".format(state_num + s1))
        print("Current Location : X : {x}, Y = {y}, Z = {z}".format(x=self.currState.location[0],
                                                                    y=self.currState.location[1],
                                                                    z=self.currState.location[2]))

        self.prevState = self.currState

        return self.currState.location, self.currState.orientation

    def _process_first_frame(self, left_frame, right_frame, state_num):

        """
        Processes first frame of the stereo state
        """

        # Initialise the initial stereo state
        self.prevState = VO_StateMachine(state_num)
        self.prevState.frames = left_frame, right_frame

        # Update the initial stereo state with detection and triangualation
        self._update_stereo_state(self.prevState)

        # Initialize the pose of the camera
        self.prevState.location = np.array(self.params.initial.location)
        self.prevState.orientation = np.array(self.params.initial.orientation)

    def _process_second_frame(self, left_frame, right_frame, state_num):

        """
        Processes second frame of the stereo state
        """

        self._process_second_frame(left_frame, right_frame, state_num)

        # Initialise the current stereo state
        self.currState = VO_StateMachine(state_num)
        self.currState.frames = left_frame, right_frame

        # Update the initial stereo state with detection and triangualation
        self._update_stereo_state(self.currState)

        # Feature Tracking from prevState to currState
        self._process_feature_tracking()

        # P3P Solver
        # obtains the pose of the camera in coordinate frame of prevState
        r_mat, t_vec = self._solve_pnp()

        # if optimisation is enabled
        # do pose updation with the optimizer
        if self.params.geometry.lsqsolver.enable:
            r_mat, t_vec = self._do_optimization(r_mat, t_vec)

        # Upating the pose of the camera of currState
        # C_n = C_n-1 * dT_n-1; where dT_n-1 is in the
        # reference of coordinate system of the second camera
        self.currState.orientation = self.prevState.orientation @ r_mat
        self.currState.location = (self.prevState.orientation @ t_vec 
                                   + self.prevState.location.reshape(-1,1)
        self.currState.location = self.currState.location.flatten()

    def _process_continuous_frame(self, left_frame, right_frame, state_num):

        """
        Processes stereo state for frames after intial processing of first 2 frames
        """

        # Initialise the current stereo state
        self.currState = VO_StateMachine(state_num)
        self.currState.frames = left_frame, right_frame

        # Feature Tracking from prevState to currState
        self._process_feature_tracking()

        # P3P Solver
        # obtains the pose of the camera in coordinate frame of prevState
        r_mat, t_vec = self._solve_pnp()

        if self.params.geometry.lsqsolver.enable:
            r_mat, t_vec = self._do_optimization(r_mat, t_vec)

        # Upating the pose of the camera of currState
        # C_n = C_n-1 * dT_n-1; where dT_n-1 is in the
        # reference of coordinate system of the second camera
        self.currState.orientation = self.prevState.orientation @ r_mat
        self.currState.location = (self.prevState.orientation @ t_vec 
                                   + self.prevState.location.reshape(-1,1)
        self.currState.location = self.currState.location.flatten()

        # Update the initial stereo state with detection and triangualation
        self._update_stereo_state(self.currState)