import copy
import numpy as np
from typing import List
from .transformation_matrix import TransformationMatrix as TM


class KinematicChain():
    def __init__(self, n_joint: int, su_joint_dict: dict,    # noqa: E999
                 bound_dict: dict, linkdh_dict: dict = None,     # noqa: E999
                 sudh_dict: dict = None, origin_poses: np.ndarray = None) -> None:
        """
        Defines a kinematic chain.
        This class enables users to easily retrieve
        Transformation Matrices to all joints and Skin Units (SU),
        and their poses (positions and orientations).

        Arguments
        -----------
        su_joint_dict: dict
            Which SU is attached to which joint.
            The dict is {i_su: i_joint}.
            where
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
        self.dof_T_dof: List[TransformationMatrix]
            Transformation Matrices BETWEEN each joint
        self.rs_T_dof: List[TransformationMatrix]
            Transformation Matrices of each joint DEFINED IN rs Frame
        self.dof_Tp_dof: List[TransformationMatrix]
            Transformation Matrices BETWEEN each joint
            given current poses :math:`\vec{\theta}`
        self.rs_Tp_dof: List[TransformationMatrix]
            Transformation Matrices of each joint DEFINED IN rs Frame
            given current poses :math:`\vec{\theta}`
        self.poses: np.ndarray
            Current Pose :math:`\vec{\theta}`
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
        self.linkdh_dict = linkdh_dict
        self.origin_poses = np.zeros(n_joint) if origin_poses is None else origin_poses

        # Construct Transformation Matrices between each joint
        self.dof_T_dof = self.__predefined_or_rand_dofs(linkdh_dict, bound_dict)
        self.dof_T_dof = self.__apply_poses(self.dof_T_dof, self.origin_poses)
        # Construct Transformation Matrices from RS to each joint
        self.rs_T_dof = self.__compute_chains_from_rs(self.dof_T_dof)

        self.origin_poses = np.zeros(n_joint) if origin_poses is None else origin_poses
        self.current_poses = np.copy(self.origin_poses)

        # Construct Transformation Matrices given poses
        # Initialize with 0 rad
        self.dof_Tp_dof = copy.deepcopy(self.dof_T_dof)
        self.rs_Tp_dof = copy.deepcopy(self.rs_T_dof)

        # Construct Transformation Matrices for each SU from its previous joint
        self.dof_T_su = self.__predefined_or_rand_sus(sudh_dict, bound_dict)

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
        dof_T_su = []
        for i in range(self.n_su):
            if sudh_dict is None:
                dof_T_vdof = TM.from_bounds(bound_dict['su'][:2, :], ['theta', 'd'])
                vdof_T_su = TM.from_bounds(bound_dict['su'][2:, :])
            else:
                dof_T_vdof = TM.from_list(sudh_dict[f'su{i+1}'][:2], ['theta', 'd'])
                vdof_T_su = TM.from_list(sudh_dict[f'su{i+1}'][2:])
            dof_T_su.append(dof_T_vdof * vdof_T_su)
        return dof_T_su

    def __compute_chains_from_rs(self, dof_T_dof: List[TM]) -> List[TM]:
        assert isinstance(dof_T_dof, list)

        rs_T_dof = []
        T = TM.from_numpy(np.zeros(4))
        for t in dof_T_dof:
            T = T * t
            rs_T_dof.append(copy.deepcopy(T))
        return rs_T_dof

    def __compute_chains_from_dof(self, i_joint: int, dof_T_dof: List[TM], rs_T_dof: List[TM]) -> List[TM]:
        """
        Unlike other functions, since this is a private function ,
        i_joint should start from 0 to n-1
        """
        assert isinstance(dof_T_dof, list)
        assert isinstance(rs_T_dof, list)
        assert len(dof_T_dof) == len(rs_T_dof)
        assert i_joint < self.n_joint

        # Start from the previous dof (or base if i_joint==0)
        T = TM.from_numpy(np.zeros(4)) if i_joint == 0 else rs_T_dof[i_joint-1]

        for i in range(i_joint, self.n_joint):
            T = T * dof_T_dof[i]
            rs_T_dof[i] = copy.deepcopy(T)
        return rs_T_dof

    def __apply_poses(self, Ts: List[TM], poses: np.ndarray) -> List[TM]:
        assert isinstance(poses, np.ndarray)
        assert len(Ts) == poses.size
        return [T(pose) for T, pose in zip(Ts, poses)]

    def reset_poses(self):
        self.current_poses = np.copy(self.origin_poses)

    def set_n_poses(self, poses: np.ndarray) -> None:
        assert isinstance(poses, np.ndarray)
        assert poses.size == self.n_joint
        self.current_poses = poses
        self.dof_Tp_dof = self.__apply_poses(self.dof_T_dof, self.current_poses)
        self.rs_Tp_dof = self.__compute_chains_from_rs(self.dof_Tp_dof)

    def add_a_pose(self, i_joint: int, pose: float) -> None:
        """
        It adds a pose to the current poses dof_Tp_dof, rs_Tp_dof

        Use this function to add 1 pose only.
        If you want to add multiple poses, use set_n_poses

        i_joint: int
            ith Joint starts from 0 to n-1
        pose: float
            Angle [rad]
        to_origin: bool
            Add theta to an origin joint if True else to a current joint
        """
        assert isinstance(i_joint, int)
        assert isinstance(pose, float)
        assert 0 <= i_joint <= self.n_joint-1, \
            print(f'i_joint Should be in between 0 and {self.n_joint-1}')

        self.current_poses[i_joint] += pose
        self.dof_Tp_dof[i_joint] = self.dof_Tp_dof[i_joint](theta=pose)
        self.rs_Tp_dof = self.__compute_chains_from_dof(i_joint, self.dof_Tp_dof, self.rs_Tp_dof)

    def __get_joint_TM(self, i_joint: int, dof_T_dof: List[TM], rs_T_dof: List[TM],
                       start_joint: int = 0) -> TM:
        """
        i_joint should also start from 0 to n-1.
        """
        assert 0 <= i_joint <= self.n_joint-1, \
            print(f'i_joint Should be in between 0 and {self.n_joint-1}')
        assert start_joint <= i_joint, \
            print(f'i_joint={i_joint} should be >= than start_joint {start_joint}')

        if start_joint == 0:
            return rs_T_dof[i_joint]

        T = dof_T_dof[start_joint]
        for i in range(start_joint+1, i_joint+1):
            T = T * dof_T_dof[i]
        return T

    def get_origin_joint_TM(self, i_joint: int, start_joint: int = 0) -> TM:
        """
        Get TransformationMatrix to the ith joint
        at when all poses are 0.
        """
        return self.__get_joint_TM(i_joint, self.dof_T_dof, self.rs_T_dof, start_joint)

    def get_current_joint_TM(self, i_joint: int, start_joint: int = 0) -> TM:
        """
        Get TransformationMatrix to the ith joint.
        at when poses are given by self.poses.
        """
        return self.__get_joint_TM(i_joint, self.dof_Tp_dof, self.rs_Tp_dof, start_joint)

    def __get_su_TM(self, i_su: int, dof_T_dof: List[TM], rs_T_dof: List[TM],
                    start_joint: int = 0) -> TM:
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

        if start_joint == 0:
            return rs_T_dof[i_joint] * self.dof_T_su[i_su]

        T = dof_T_dof[start_joint]
        for j in range(start_joint+1, i_joint+1):
            T = T * dof_T_dof[j]
        return T * self.dof_T_su[i_su]

    def get_origin_su_TM(self, i_su: int, start_joint: int = 0) -> TM:
        """
        Get TransformationMatrix to the ith su
        at when all poses are 0.
        """
        return self.__get_su_TM(i_su, self.dof_T_dof, self.rs_T_dof, start_joint)

    def get_current_su_TM(self, i_su: int, start_joint: int = 0) -> TM:
        """
        Get TransformationMatrix to the ith su.
        at when poses are given by self.poses.
        """
        return self.__get_su_TM(i_su, self.dof_Tp_dof, self.rs_Tp_dof, start_joint)

    def set_sudh(self, i_su: int, params: np.ndarray) -> None:
        """
        Sets i_su th SU DH Parameters

        Parameters
        -----------
        i_su: int
            i_su th SU. i_su starts from 0 to m-1.
        params: np.ndarray
            DH Parameters of the i_su th SU (from its previous DoF)
        """
        assert 0 <= i_su <= self.n_su-1
        assert params.size == 6

        dof_T_vdof = TM.from_numpy(params[:2], ['theta', 'd'])
        vdof_T_su = TM.from_numpy(params[2:])
        self.dof_T_su[i_su] = dof_T_vdof * vdof_T_su

    def set_linkdh(self, i_joint: int, params: np.ndarray) -> None:
        """
        Sets i_su th SU DH Parameters

        Parameters
        -----------
        i_joint: int
            i_joint th joint. i_joint starts from 0 to n-1.
        params: np.ndarray
            DH Parameters of the i_joint th joint (from its previous joint)
        """
        assert 0 <= i_joint <= self.n_joint-1
        assert params.size == 4

        self.dof_T_dof[i_joint] = TM.from_numpy(params)(self.origin_poses[i_joint])
        self.rs_T_dof = self.__compute_chains_from_dof(i_joint, self.dof_T_dof, self.rs_T_dof)
        self.dof_Tp_dof[i_joint] = TM.from_numpy(params)(self.current_poses[i_joint])
        self.rs_Tp_dof = self.__compute_chains_from_dof(i_joint, self.dof_Tp_dof, self.rs_Tp_dof)
