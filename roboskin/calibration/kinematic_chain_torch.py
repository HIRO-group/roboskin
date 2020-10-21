import copy
import numpy as np
from roboskin.calibration.transformation_matrix import TransformationMatrix as TM
import torch

BOUNDS = np.array([
    [-np.pi, np.pi],    # th
    [-1.0, 1.0],        # d
    [-1.0, 1.0],        # a     (radius)
    [-np.pi, np.pi]])   # alpha
BOUNDS_SU = np.array([
    [-np.pi, np.pi],    # th
    [-1.0, 1.0],        # d
    [-np.pi, np.pi],    # th
    [-1.0, 1.0],        # d
    [-1.0, 1.0],        # a     # 0 gives error
    [-np.pi, np.pi]])   # alpha


def construct_kinematic_chain(robot_configs, imu_mappings,
                              test_code=False, optimize_all=False):
    """
    Arguments
    ----------
    robot_configs: dict
    imu_mappings: dict
    test_code: bool
    optimize_all: bool
    """
    su_joint_dict = {}
    joints = []
    for imu_str, link_str in imu_mappings.items():
        su_joint_dict[int(imu_str[-1])] = int(link_str[-1]) - 1
        joints.append(int(link_str[-1]) - 1)
    joints = np.unique(joints)
    bound_dict = {'link': BOUNDS, 'su': BOUNDS_SU}
    keys = ['dh_parameter', 'su_dh_parameter', 'eval_poses']
    for key in keys:
        if key not in robot_configs:
            raise KeyError('Keys {} should exist in robot yaml file'.format(keys))
    linkdh_dict = None if optimize_all else robot_configs['dh_parameter']
    sudh_dict = robot_configs['su_dh_parameter'] if test_code else None
    eval_poses = np.array(robot_configs['eval_poses'])
    kinematic_chain = KinematicChainTorch(
        n_joint=joints.size,
        su_joint_dict=su_joint_dict,
        bound_dict=bound_dict,
        linkdh_dict=linkdh_dict,
        sudh_dict=sudh_dict,
        eval_poses=eval_poses)
    if optimize_all:
        linkdh0 = np.array(robot_configs['dh_parameter']['joint1'])
        su0 = np.random.rand(6)
        params = np.r_[linkdh0, su0]
        kinematic_chain.set_params_at(0, params)
    return kinematic_chain


class KinematicChainTorch():
    def __init__(self, n_joint, su_joint_dict, bound_dict, linkdh_dict=None,
                 sudh_dict=None, eval_poses=None):
        """
        Defines a kinematic chain.
        This class enables users to easily retrieve
        Transformation Matrices to all joints and Skin Units (SU),
        and their poses (positions and orientations).

        :math:`dof_T*_dof` represents a list of Transformation Matrices
        between each joint
        :math:`rs_T*_dof` represents a list of Transformation Matrices
        from RS frame to each joint

        Arguments
        -----------
        su_joint_dict: dict
            Which SU is attached to which joint.
            The dict is {i_su: i_joint} where

            ..math::
                i_su = 0, ..., n_su-1
                i_joint = 0, ..., n_joint-1

        n_joint: int
            number of joints
        bound_dict: dict
            Bounds of DH Parameters
            {'link': np.ndarray (4, 2), 'su': np.ndarray (6, 2)}
        linkdh_dict: dict
        sudh_dict: dict
        eval_poses: np.ndarray
            DH Parameters of all links.

        Attributes
        ------------
        self.dof_T0_dof: List[TransformationMatrix]
            Transformation Matices betw. joints at Origin Poses
        self.dof_Tc_dof: List[TransformationMatrix]
            Transformation Matices betw. joints at Current Poses
        self.rs_T0_dof: List[TransformationMatrix]
            Transformation Matices from RS to each joint at Origin Poses
        self.rs_Tc_dof: List[TransformationMatrix]
            Transformation Matices from RS to each joint at Current Poses
        self.eval_poses: np.ndarray
            Joint Poses at when each SU pose is evaluated.
        self.current_poses: np.ndarray
            Current Pose :math:`\vec{\theta}`
        self.dof_T_vdof:
            Transformation Mtarices from each joint to SU's virtual joint
        self.vdof_T_su:
            Transformation Mtarices from a virtual joint to its SU
        self.dof_T_su:
            Transformation Mtarices from each joint to its SU
            Multiplication of self.dof_T_vdof and self.vdof_T_su
        """
        assert isinstance(n_joint, int)
        assert isinstance(su_joint_dict, dict)
        assert isinstance(bound_dict, dict)
        assert len(su_joint_dict) != 0
        assert 'link' in list(bound_dict.keys())
        assert 'su' in list(bound_dict.keys())
        assert bound_dict['link'].shape == (4, 2)
        assert bound_dict['su'].shape == (6, 2)
        if linkdh_dict is not None:
            assert isinstance(linkdh_dict, dict)
            assert len(linkdh_dict) == n_joint
        if sudh_dict is not None:
            assert isinstance(sudh_dict, dict)
            assert len(sudh_dict) == len(su_joint_dict)
        self.su_joint_dict = su_joint_dict
        self.n_su = len(su_joint_dict)
        self.n_joint = n_joint
        self.bound_dict = bound_dict
        self.linkdh_dict = linkdh_dict
        self.sudh_dict = sudh_dict
        self.eval_poses = torch.zeros(n_joint).cuda() if eval_poses is None else eval_poses
        self.current_poses = torch.zeros(self.n_joint).cuda()

        # At Original Poses (joints == 0 rad)
        self.dof_T0_dof = self.__predefined_or_rand_dofs(linkdh_dict, bound_dict)
        self.rs_T0_dof = self.__initialize_chains(self.dof_T0_dof)
        self.rs_Te_dof = self._copy_transmat_torch(self.rs_T0_dof)

        self.dof_Te_dof = self.__apply_poses(self.eval_poses, self.dof_T0_dof, self.rs_Te_dof)
        # At CURRENT pose (joints == current_poses)
        self.dof_Tc_dof = self._copy_transmat_torch(self.dof_T0_dof)
        self.rs_Tc_dof = self._copy_transmat_torch(self.rs_T0_dof)

        self.temp_poses = None
        self.dof_Tt_dof = None
        self.rs_Tt_dof = None

        # Construct Transformation Matrices for each SU from its previous joint
        self.dof_T_vdof, self.vdof_T_su, self.dof_T_su = \
            self.predefined_or_rand_sus(sudh_dict, bound_dict)

    def _copy_transmat_torch(self, transmats):
        """
        copy transformation matrix in torch. Requires
        a few extra steps.
        """
        shallow_copy = copy.copy(transmats)
        for t in shallow_copy:
            t.matrix = t.matrix.clone()
            t.params = t.params.clone()
            t.q = copy.deepcopy(t.q)
        return shallow_copy

    def __predefined_or_rand_dofs(self, linkdh_dict, bound_dict):
        """
        Arguments
        ---------
        linkdh_dict: dict
        bound_dict: dict

        Returns
        --------
        : List[TM]
        """
        if linkdh_dict is None:
            # Initialize DH parameters randomly within the given bounds
            return [TM.from_bounds(bound_dict['link']).tensor()
                    for i in range(self.n_joint)]
        else:
            # Specified DH Parameters
            return [TM.from_list(linkdh_dict['joint{}'.format(i+1)]).tensor()
                    for i in range(self.n_joint)]

    def __predefined_or_rand_sus(self, sudh_dict, bound_dict):
        """
        Arguments
        ---------
        sudh_dict: dict
        bound_dict: dict

        Returns
        --------
        : List[TM]
        """
        dof_T_vdof = []
        vdof_T_su = []
        dof_T_su = []
        for i in range(self.n_su):
            if sudh_dict is None:
                _dof_T_vdof = TM.from_bounds(bound_dict['su'][:2, :], ['theta', 'd']).tensor()
                _vdof_T_su = TM.from_bounds(bound_dict['su'][2:, :]).tensor()
            else:
                _dof_T_vdof = TM.from_list(sudh_dict['su{}'.format(i+1)][:2], ['theta', 'd']).tensor()
                _vdof_T_su = TM.from_list(sudh_dict['su{}'.format(i+1)][2:]).tensor()
            dof_T_vdof.append(_dof_T_vdof)
            vdof_T_su.append(_vdof_T_su)
            dof_T_su.append(_dof_T_vdof * _vdof_T_su)
        return dof_T_vdof, vdof_T_su, dof_T_su

    def __initialize_chains(self, dof_T_dof):
        """
        initialize chains (rs to dof matrices) from
        dof to dof matrices.

        Arguments
        ----------
        dof_T_dof: List[TM]

        Returns
        --------
        : List[TM]
        """
        start_joint = 0
        # list of nones - to be filled in later
        rs_T_dof = [None]*self.n_joint
        self.__update_chains(dof_T_dof, rs_T_dof, start_joint)
        return rs_T_dof

    def __update_chains(self, dof_T_dof, rs_T_dof, start_joint=0, end_joint=None):
        """
        Updates rs to dof matrices based on dof to dof matrices.

        Unlike other functions, since this is a private function.
        i_joint should start from 0 to n-1

        Arguments
        ----------
        dof_T_dof: List[TM]
        rs_T_dof: List[TM]
        start_joint: int
        end_joint: int

        Returns
        --------
        : List[TM]
        """
        assert isinstance(dof_T_dof, list)
        assert isinstance(rs_T_dof, list)
        assert len(dof_T_dof) == len(rs_T_dof)
        if end_joint is None:
            end_joint = self.n_joint - 1

        # Start from the previous DoF (or base if i_joint==0)
        T = TM.from_numpy(np.zeros(4)).tensor() if start_joint == 0 else rs_T_dof[start_joint-1]

        # construct reference segment to dof matrix from dof to dof matrix.
        for i in range(start_joint, end_joint+1):
            T = T * dof_T_dof[i]
            rs_T_dof[i] = T

    def __apply_poses(self, poses, dof_T_dof, rs_T_dof, start_joint=0, end_joint=None):
        """
        based on `poses` and specified start and end joints,
        applies a pose.

        Arguments
        ----------
        poses: np.ndarray
        dof_T_dof: List[TM]
        rs_T_dof: List[TM]
        start_joint: int
        end_joint: int

        Returns
        --------
        : List[TM]
        """
        poses = torch.tensor(poses).cuda()
        assert isinstance(poses, torch.Tensor)
        assert len(dof_T_dof) == len(poses)
        if end_joint is None:
            end_joint = self.n_joint - 1

        # Start from the previous DoF (or base if i_joint==0)
        T = TM.from_numpy(np.zeros(4)).tensor() if start_joint == 0 else rs_T_dof[start_joint-1]

        dof_Tc_dof = self._copy_transmat_torch(dof_T_dof)

        # dof_Tc_dof = copy.deepcopy(dof_T_dof)
        for i in range(start_joint, end_joint+1):
            dof_Tc_dof[i] = dof_T_dof[i](poses[i])
            T = T * dof_Tc_dof[i]
            rs_T_dof[i] = T
        return dof_Tc_dof

    def reset_poses(self):
        """
        Resets current and temporary poses to 0s.
        Origin and Evaluation Poses will never be changed.
        """
        self.current_poses = torch.zeros(self.n_joint).cuda()
        self.dof_Tc_dof = self._copy_transmat_torch(self.dof_T0_dof)
        self.rs_Tc_dof = self._copy_transmat_torch(self.rs_T0_dof)
        # self.dof_Tc_dof = copy.deepcopy(self.dof_T0_dof)
        # self.rs_Tc_dof = copy.deepcopy(self.rs_T0_dof)

    def set_poses(self, poses, start_joint=0, end_joint=None):
        """
        sets poses from start_joint to end_joint.

        Arguments
        ----------
        poses: np.ndarray
        start_joint: int
        end_joint: int
        """
        poses = torch.tensor(poses).cuda()
        assert isinstance(poses, torch.Tensor)
        assert poses.shape[0] == self.n_joint
        if end_joint is None:
            end_joint = self.n_joint - 1
        # Set current pose
        self.current_poses[start_joint:end_joint+1] = poses[start_joint:end_joint+1]
        # Compute dof_Tc_dof and update rs_Tc_dof
        self.dof_Tc_dof = self.__apply_poses(
            self.current_poses, self.dof_T0_dof, self.rs_Tc_dof, start_joint, end_joint)

        self.dof_Tt_dof = None
        self.rs_Tt_dof = None
        self.temp_poses = None

    def init_temp_TM(self, i_joint, additional_pose):
        """
        Initialize a temporary Transformation Matrices by adding
        an extra joint angle to Current Tranformation Matrices.
        Current Pose will not be updated.
        The temporary pose/TM will be reset once init_temp_TM or set_poses are called.

        TODO: Allow multiple additional poses

        Parameters
        -----------
        `i_joint`: `int`
            i_joint th joint
        `additional_pose`: `float`
            Additional pose added to the pose (copied from the current pose).
            Current pose will not be updated.
        """

        self.temp_poses = copy.deepcopy(self.current_poses)
        self.dof_Tt_dof = self._copy_transmat_torch(self.dof_Tc_dof)
        self.rs_Tt_dof = self._copy_transmat_torch(self.rs_Tc_dof)

        # Update current poses and copy them to temporary poses
        self.temp_poses[i_joint] += additional_pose
        self.dof_Tt_dof[i_joint] = self.dof_Tt_dof[i_joint](theta=additional_pose)
        self.__update_chains(self.dof_Tt_dof, self.rs_Tt_dof, start_joint=i_joint)

    def add_temp_pose(self, i_joint, additional_pose):
        """
        Add a pose to the temporary pose/TM
        It does not initialize the temporary pose
        nor overwrite the current pose.
        The temporary pose/TM will be reset once init_temp_TM or set_poses are called.

        TODO: Allow multiple additional poses

        Parameters
        -----------
        i_joint: int
        additional_pose:float
        """
        self.temp_poses[i_joint] += additional_pose
        self.dof_Tt_dof[i_joint] = self.dof_Tt_dof[i_joint](theta=additional_pose)
        self.__update_chains(self.dof_Tt_dof, self.rs_Tt_dof, start_joint=i_joint)

    def __compute_joint_TM(self, i_joint, dof_T_dof, rs_T_dof, start_joint):
        """
        i_joint should also start from 0 to n-1.

        Parameters
        -----------
        i_joint: int
        dof_T_dof: List[TM]
        rs_T_dof: List[TM]
        start_joint: int

        Returns
        -------
        : TM
        """
        assert 0 <= i_joint <= self.n_joint-1  #, print('i_joint Should be in between 0 and {}'.format(self.n_joint-1))
        assert start_joint <= i_joint  #, print('i_joint={} should be >= than start_joint {}'.format(i_joint, start_joint))

        if start_joint == -1:
            return rs_T_dof[i_joint]

        T = dof_T_dof[start_joint]
        # compute transformations up to end joint.
        for i in range(start_joint+1, i_joint+1):
            T = T * dof_T_dof[i]
        return T

    def compute_joint_TM(self, i_joint, pose_type, start_joint=-1):
        """
        Get a TransformationMatrix to the i_joint th joint

        Based on `pose_type`, computes transformation matrix to joint.

        Parameters
        -----------
        i_joint: int
        pose_type: str
        start_joint: int

        Returns
        -------
        : TM
        """
        if pose_type == 'orgin':
            return self.__compute_joint_TM(i_joint, self.dof_T0_dof, self.rs_T0_dof, start_joint)
        if pose_type == 'eval':
            return self.__compute_joint_TM(i_joint, self.dof_Te_dof, self.rs_Te_dof, start_joint)
        if pose_type == 'current':
            return self.__compute_joint_TM(i_joint, self.dof_Tc_dof, self.rs_Tc_dof, start_joint)
        elif pose_type == 'temp':
            if self.dof_Tt_dof is None or self.rs_Tt_dof is None:
                raise ValueError('Temprary Pose is not set')
            return self.__compute_joint_TM(i_joint, self.dof_Tt_dof, self.rs_Tt_dof, start_joint)
        else:
            raise ValueError('Not such pose as {}'.format(pose_type))

    def __compute_su_TM(self, i_su, dof_T_dof, rs_T_dof, start_joint):
        """
        i_su should also start from 0 to m-1.

        Parameters
        -----------
        i_su: int
        dof_T_dof: List[TM]
        rs_T_dof: List[TM]
        start_joint: int

        Returns
        --------
        : TM
        """
        assert 0 <= i_su <= self.n_su-1  #, print('i_su Should be in between 0 and {}'.format(self.n_su-1))

        # Get corresponding joint number
        i_joint = self.su_joint_dict[i_su]

        assert start_joint <= i_joint  #, print('i_joint {} which i_su {} is attached to should be larger than or equal to start_joint {}'.format(i_joint, i_su, start_joint))

        if start_joint == -1:
            return rs_T_dof[i_joint] * self.dof_T_su[i_su]

        T = TM.from_numpy(np.zeros(4)).tensor()
        for j in range(start_joint+1, i_joint+1):
            T = T * dof_T_dof[j]
        return T * self.dof_T_su[i_su]

    def compute_su_TM(self, i_su, pose_type, start_joint=-1):
        """
        Get a TransformationMatrix to the i_su th su

        Arguments
        ----------
        i_su: int
        pose_type: str
        start_joint: int

        Returns
        ---------
        : TM
        """
        if pose_type == 'origin':
            return self.__compute_su_TM(i_su, self.dof_T0_dof, self.rs_T0_dof, start_joint)
        if pose_type == 'eval':
            return self.__compute_su_TM(i_su, self.dof_Te_dof, self.rs_Te_dof, start_joint)
        if pose_type == 'current':
            return self.__compute_su_TM(i_su, self.dof_Tc_dof, self.rs_Tc_dof, start_joint)
        elif pose_type == 'temp':
            if self.dof_Tt_dof is None or self.rs_Tt_dof is None:
                raise ValueError('Temprary Pose is not set')
            return self.__compute_su_TM(i_su, self.dof_Tt_dof, self.rs_Tt_dof, start_joint)
        else:
            raise ValueError('There is no such pose_type as {}'.format(pose_type))

    def set_sudh(self, i_su, params):
        """
        Set i_su th SU DH Parameters

        Parameters
        -----------
        i_su: int
            i_su th SU. i_su starts from 0 to m-1.
        params: np.ndarray
            DH Parameters of the i_su th SU (from its previous DoF)
        """
        assert 0 <= i_su <= self.n_su-1
        assert params.shape[0] == 6

        self.dof_T_vdof[i_su] = TM.from_numpy(params[:2], ['theta', 'd']).tensor()
        self.vdof_T_su[i_su] = TM.from_numpy(params[2:]).tensor()
        self.dof_T_su[i_su] = self.dof_T_vdof[i_su] * self.vdof_T_su[i_su]

    def set_linkdh(self, i_joint, params):
        """
        Set i_su th SU DH Parameters

        Parameters
        -----------
        i_joint: int
            i_joint th joint. i_joint starts from 0 to n-1.
        params: np.ndarray
            DH Parameters of the i_joint th joint (from its previous joint)
        """
        assert 0 <= i_joint <= self.n_joint-1
        assert params.size == 4

        self.dof_T0_dof[i_joint] = TM.from_numpy(params).tensor()
        self.__update_chains(self.dof_T0_dof, self.rs_T0_dof, start_joint=i_joint)
        self.dof_Tc_dof[i_joint] = TM.from_numpy(params).tensor()
        self.rs_Tc_dof = self._copy_transmat_torch(self.rs_T0_dof)
        self.rs_Te_dof = self._copy_transmat_torch(self.rs_T0_dof)

        self.dof_Te_dof = self.__apply_poses(self.eval_poses, self.dof_T0_dof, self.rs_Te_dof)

    def get_params_at(self, i_su):
        """

        Arguments
        ---------------
        i_su: int
            ith SU

        Returns
        --------
        params: torch.Tensor
            Next DH parameters to be optimized
        bounds: torch.Tensor
            Bounds of each DH parameter
        """
        i_joint = self.su_joint_dict[i_su]

        if self.linkdh_dict is None:

            # optimizing all dh parameters
            params = torch.cat((self.dof_T0_dof[i_joint].parameters,  # 4
                                self.dof_T_vdof[i_su].parameters,  # 2
                                self.vdof_T_su[i_su].parameters))  # 4
            bounds = torch.stack((self.bound_dict['link'],
                                  self.bound_dict['su']))
            assert params.size == 10
            assert bounds.shape == (10, 2)
        else:
            # optimizing just su dh params.
            params = torch.cat((self.dof_T_vdof[i_su].parameters,  # 2
                                self.vdof_T_su[i_su].parameters))  # 4

            bounds = self.bound_dict['su']
            assert params.shape[0] == 6
            assert bounds.shape == (6, 2)

        return params, bounds

    def set_params_at(self, i_su, params):
        """
        Set DH parameters
        Depending of if we
        are optimizing 6 (just su params)
        or 10 (all dh params)

        Arguments
        ------------
        int: i
            ith joint (sensor)
        parmas: np.array
            DH Parameters
        """
        if type(params) != torch.Tensor:
            params = torch.tensor(params).cuda()
        i_joint = self.su_joint_dict[i_su]

        if self.linkdh_dict is None:
            self.set_linkdh(i_joint, params[:4])
            if self.sudh_dict is None:
                self.set_sudh(i_su, params[4:])
        else:
            if self.sudh_dict is None:
                self.set_sudh(i_su, params)
