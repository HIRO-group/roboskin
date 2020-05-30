import copy
import numpy as np
from typing import List
from .transformation_matrix import TransformationMatrix as TM

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


def construct_kinematic_chain(robot_configs: dict, imu_mappings: dict,
                              test_code=False, optimize_all=False):
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
            raise KeyError(f'Keys {keys} should exist in robot yaml file')

    linkdh_dict = None if optimize_all else robot_configs['dh_parameter']
    sudh_dict = robot_configs['su_dh_parameter'] if test_code else None
    eval_poses = np.array(robot_configs['eval_poses'])

    kinematic_chain = KinematicChain(
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


class KinematicChain():
    def __init__(self, n_joint: int, su_joint_dict: dict,  # noqa: E999
                 bound_dict: dict, linkdh_dict: dict = None,     # noqa: E999
                 sudh_dict: dict = None, eval_poses: np.ndarray = None) -> None:
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
        self.eval_poses = np.zeros(n_joint) if eval_poses is None else eval_poses
        self.current_poses = np.zeros(self.n_joint)

        # At Original Poses (joints == 0 rad)
        self.dof_T0_dof = self.__predefined_or_rand_dofs(linkdh_dict, bound_dict)
        self.rs_T0_dof = self.__initialize_chains(self.dof_T0_dof)

        self.rs_Te_dof = copy.deepcopy(self.rs_T0_dof)
        self.dof_Te_dof = self.__apply_poses(self.eval_poses, self.dof_T0_dof, self.rs_Te_dof)

        # At CURRENT pose (joints == current_poses)
        self.dof_Tc_dof = copy.deepcopy(self.dof_T0_dof)
        self.rs_Tc_dof = copy.deepcopy(self.rs_T0_dof)

        self.temp_poses = None
        self.dof_Tt_dof = None
        self.rs_Tt_dof = None

        # Construct Transformation Matrices for each SU from its previous joint
        self.dof_T_vdof, self.vdof_T_su, self.dof_T_su = \
            self.__predefined_or_rand_sus(sudh_dict, bound_dict)

    def __predefined_or_rand_dofs(self, linkdh_dict: dict, bound_dict: dict) -> List[TM]:
        if linkdh_dict is None:
            # Initialize DH parameters randomly within the given bounds
            return [TM.from_bounds(bound_dict['link'])
                    for i in range(self.n_joint)]
        else:
            # Specified DH Parameters
            return [TM.from_list(linkdh_dict[f'joint{i+1}'])
                    for i in range(self.n_joint)]

    def __predefined_or_rand_sus(self, sudh_dict: dict, bound_dict: dict) -> List[TM]:
        dof_T_vdof = []
        vdof_T_su = []
        dof_T_su = []
        for i in range(self.n_su):
            if sudh_dict is None:
                _dof_T_vdof = TM.from_bounds(bound_dict['su'][:2, :], ['theta', 'd'])
                _vdof_T_su = TM.from_bounds(bound_dict['su'][2:, :])
            else:
                _dof_T_vdof = TM.from_list(sudh_dict[f'su{i+1}'][:2], ['theta', 'd'])
                _vdof_T_su = TM.from_list(sudh_dict[f'su{i+1}'][2:])
            dof_T_vdof.append(_dof_T_vdof)
            vdof_T_su.append(_vdof_T_su)
            dof_T_su.append(_dof_T_vdof * _vdof_T_su)
        return dof_T_vdof, vdof_T_su, dof_T_su

    def __initialize_chains(self, dof_T_dof: List[TM]) -> List[TM]:
        start_joint = 0
        rs_T_dof = [None]*self.n_joint
        self.__update_chains(dof_T_dof, rs_T_dof, start_joint)
        return rs_T_dof

    def __update_chains(self, dof_T_dof: List[TM], rs_T_dof: List[TM],
                        start_joint: int = 0, end_joint: int = None):
        """
        Unlike other functions, since this is a private function ,
        i_joint should start from 0 to n-1
        """
        assert isinstance(dof_T_dof, list)
        assert isinstance(rs_T_dof, list)
        assert len(dof_T_dof) == len(rs_T_dof)
        if end_joint is None:
            end_joint = self.n_joint - 1

        # Start from the previous DoF (or base if i_joint==0)
        T = TM.from_numpy(np.zeros(4)) if start_joint == 0 else rs_T_dof[start_joint-1]

        for i in range(start_joint, end_joint+1):
            T = T * dof_T_dof[i]
            rs_T_dof[i] = T

    def __apply_poses(self, poses: np.ndarray, dof_T_dof: List[TM], rs_T_dof: List[TM],
                      start_joint: int = 0, end_joint: int = None) -> List[TM]:
        assert isinstance(poses, np.ndarray)
        assert len(dof_T_dof) == poses.size
        if end_joint is None:
            end_joint = self.n_joint - 1

        # Start from the previous DoF (or base if i_joint==0)
        T = TM.from_numpy(np.zeros(4)) if start_joint == 0 else rs_T_dof[start_joint-1]

        dof_Tc_dof = copy.deepcopy(dof_T_dof)
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
        self.current_poses = np.zeros(self.n_joint)
        self.dof_Tc_dof = copy.deepcopy(self.dof_T0_dof)
        self.rs_Tc_dof = copy.deepcopy(self.rs_T0_dof)

    def set_poses(self, poses: np.ndarray,
                  start_joint: int = 0, end_joint: int = None) -> None:
        """
        Set Current Poses.
        """
        assert isinstance(poses, np.ndarray)
        assert poses.size == self.n_joint
        if end_joint is None:
            end_joint = self.n_joint - 1
        # Set current pose
        self.current_poses[start_joint:end_joint+1] = poses[start_joint:end_joint+1]
        # Compute dof_Tc_dof and update rs_Tc_dof
        self.dof_Tc_dof = self.__apply_poses(
            self.current_poses, self.dof_T0_dof, self.rs_Tc_dof, start_joint, end_joint)
        # Set temporary Ts to None
        self.dof_Tt_dof = None
        self.rs_Tt_dof = None
        self.temp_poses = None

    def init_temp_TM(self, i_joint: int, additional_pose: float) -> None:
        """
        Initialize a temporary Transformation Matrices by adding
        an extra joint angle to Current Tranformation Matrices.
        Current Pose will not be updated.
        """
        self.temp_poses = copy.deepcopy(self.current_poses)
        self.dof_Tt_dof = copy.deepcopy(self.dof_Tc_dof)
        self.rs_Tt_dof = copy.deepcopy(self.rs_Tc_dof)

        # Update current poses and copy them to temporary poses
        self.temp_poses[i_joint] += additional_pose
        self.dof_Tt_dof[i_joint] = self.dof_Tt_dof[i_joint](theta=additional_pose)
        self.__update_chains(self.dof_Tt_dof, self.rs_Tt_dof, start_joint=i_joint)

    def add_temp_pose(self, i_joint: int, additional_pose: float) -> None:
        self.temp_poses[i_joint] += additional_pose
        self.dof_Tt_dof[i_joint] = self.dof_Tt_dof[i_joint](theta=additional_pose)
        self.__update_chains(self.dof_Tt_dof, self.rs_Tt_dof, start_joint=i_joint)

    def __compute_joint_TM(self, i_joint: int, dof_T_dof: List[TM], rs_T_dof: List[TM],
                          start_joint: int) -> TM:
        """
        i_joint should also start from 0 to n-1.
        """
        assert 0 <= i_joint <= self.n_joint-1, \
            print(f'i_joint Should be in between 0 and {self.n_joint-1}')
        assert start_joint <= i_joint, \
            print(f'i_joint={i_joint} should be >= than start_joint {start_joint}')

        if start_joint == -1:
            return rs_T_dof[i_joint]

        T = dof_T_dof[start_joint]
        for i in range(start_joint+1, i_joint+1):
            T = T * dof_T_dof[i]
        return T

    def compute_joint_TM(self, i_joint: int, pose_type: str, start_joint: int = -1) -> TM:
        """
        Get a TransformationMatrix to the i_joint th joint
        """
        if pose_type == 'orgin':
            return self.__compute_joint_TM(i_joint, self.dof_T0_dof, self.rs_T0_dof, start_joint)
        elif pose_type == 'eval':
            return self.__compute_joint_TM(i_joint, self.dof_Te_dof, self.rs_Te_dof, start_joint)
        elif pose_type == 'current':
            return self.__compute_joint_TM(i_joint, self.dof_Tc_dof, self.rs_Tc_dof, start_joint)
        elif pose_type == 'temp':
            if self.dof_Tt_dof is None or self.rs_Tt_dof is None:
                raise ValueError('Temprary Pose is not set')
            return self.__compute_joint_TM(i_joint, self.dof_Tt_dof, self.rs_Tt_dof, start_joint)
        else:
            raise ValueError(f'Not such pose as {pose_type}')

    def __compute_su_TM(self, i_su: int, dof_T_dof: List[TM], rs_T_dof: List[TM],
                       start_joint: int) -> TM:
        """
        i_su should also start from 0 to m-1.
        """
        assert 0 <= i_su <= self.n_su-1, \
            print(f'i_su Should be in between 0 and {self.n_su-1}')

        # Get corresponding joint number
        i_joint = self.su_joint_dict[i_su]

        assert start_joint <= i_joint, \
            print(f'i_joint {i_joint} which i_su {i_su} is attached to \
                    should be larger than or equal to start_joint {start_joint}')

        if start_joint == -1:
            return rs_T_dof[i_joint] * self.dof_T_su[i_su]

        T = TM.from_numpy(np.zeros(4))
        for j in range(start_joint+1, i_joint+1):
            T = T * dof_T_dof[j]
        return T * self.dof_T_su[i_su]

    def compute_su_TM(self, i_su: int, pose_type: str, start_joint: int = -1) -> TM:
        """
        Get a TransformationMatrix to the i_su th su
        """
        if pose_type == 'origin':
            return self.__compute_su_TM(i_su, self.dof_T0_dof, self.rs_T0_dof, start_joint)
        elif pose_type == 'eval':
            return self.__compute_su_TM(i_su, self.dof_Te_dof, self.rs_Te_dof, start_joint)
        elif pose_type == 'current':
            return self.__compute_su_TM(i_su, self.dof_Tc_dof, self.rs_Tc_dof, start_joint)
        elif pose_type == 'temp':
            if self.dof_Tt_dof is None or self.rs_Tt_dof is None:
                raise ValueError('Temprary Pose is not set')
            return self.__compute_su_TM(i_su, self.dof_Tt_dof, self.rs_Tt_dof, start_joint)
        else:
            raise ValueError(f'There is no such pose_type as {pose_type}')

    def set_sudh(self, i_su: int, params: np.ndarray) -> None:
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
        assert params.size == 6

        self.dof_T_vdof[i_su] = TM.from_numpy(params[:2], ['theta', 'd'])
        self.vdof_T_su[i_su] = TM.from_numpy(params[2:])
        self.dof_T_su[i_su] = self.dof_T_vdof[i_su] * self.vdof_T_su[i_su]

    def set_linkdh(self, i_joint: int, params: np.ndarray) -> None:
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

        self.dof_T0_dof[i_joint] = TM.from_numpy(params)
        self.__update_chains(self.dof_T0_dof, self.rs_T0_dof, start_joint=i_joint)
        self.dof_Tc_dof[i_joint] = TM.from_numpy(params)
        self.rs_Tc_dof = copy.deepcopy(self.rs_T0_dof)
        self.rs_Te_dof = copy.deepcopy(self.rs_T0_dof)
        self.dof_Te_dof = self.__apply_poses(self.eval_poses, self.dof_T0_dof, self.rs_Te_dof)

    def get_params_at(self, i_su: int):
        """

        Arguments
        ---------------
        i_su: int
            ith SU

        Returns
        --------
        params: np.array
            Next DH parameters to be optimized
        bounds: np.array
            Bounds of each DH parameter
        """
        i_joint = self.su_joint_dict[i_su]

        if self.linkdh_dict is None:
            # optimizing all dh parameters
            params = np.r_[self.dof_T0_dof[i_joint].parameters,     # 4
                           self.dof_T_vdof[i_su].parameters,        # 2
                           self.vdof_T_su[i_su].parameters]         # 4
            bounds = np.vstack((self.bound_dict['link'],
                                self.bound_dict['su']))
            assert params.size == 10
            assert bounds.shape == (10, 2)
        else:
            # optimizing just su dh params.
            params = np.r_[self.dof_T_vdof[i_su].parameters,    # 2
                           self.vdof_T_su[i_su].parameters]     # 4
            bounds = self.bound_dict['su']
            assert params.size == 6
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
        i_joint = self.su_joint_dict[i_su]

        if self.linkdh_dict is None:
            self.set_linkdh(i_joint, params[:4])
            if self.sudh_dict is None:
                self.set_sudh(i_su, params[4:])
        else:
            if self.sudh_dict is None:
                self.set_sudh(i_su, params)
