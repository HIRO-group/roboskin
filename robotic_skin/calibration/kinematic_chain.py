import copy
import numpy as np
from .transformation_matrix import TransformationMatrix as TM


class KinematicChain():
    def __init__(self, n_joint: int, su_dict: dict,
                 bound_dict: dict, linkdh_dict: dict = None) -> None:
        """
        Defines a kinematic chain.
        This class enables users to easily retrieve
        Transformation Matrices to all joints and Skin Units (SU),
        and their poses (positions and orientations).

        Arguments
        -----------
        su_dict: dict
            Which SU is attached to which joint.
            The dict is {i_su: i_joint}.
            where
            ..math::
                i_su = 1, ..., n_su
                i_joint = 1, ..., n_joint
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
        assert isinstance(su_dict, dict)
        assert isinstance(bound_dict, dict)
        assert isinstance(linkdh_dict, dict)
        assert len(su_dict) != 0
        assert 'link' in list(bound_dict.keys())
        assert 'su' in list(bound_dict.keys())
        assert bound_dict['link'].shape == (4, 2)
        assert bound_dict['su'].shape == (6, 2)
        if linkdh_dict is not None:
            assert len(linkdh_dict) == n_joint

        self.su_dict = su_dict
        self.n_su = len(su_dict)
        self.n_joint = n_joint
        self.linkdh_dict = linkdh_dict

        # Construct Transformation Matrices between each joint
        if self.linkdh_dict is None:
            # Initialize DH parameters randomly within the given bounds
            self.dof_T_dof = [TM.from_bounds(bound_dict['link'])
                              for i in range(n_joint)]
        else:
            # Specified DH Parameters
            self.dof_T_dof = [TM.from_list(linkdh_dict['joint%i' % (i+1)])
                              for i in range(n_joint)]

        # Construct Transformation Matrices from RS to each joint
        self.rs_T_dof = self.__compute_chains_from_rs(self.dof_T_dof)

        # Construct Transformation Matrices given poses
        # Initialize with 0 rad which is equal to dof_T_dof, su_T_dof
        self.poses = np.zeros(self.n_joint)
        self.dof_Tp_dof = self.__apply_poses(self.dof_T_dof, self.poses)
        self.rs_Tp_dof = self.__compute_chains_from_rs(self.dof_Tp_dof)

        # Construct Transformation Matrices
        self.dof_T_vdof = []
        self.vdof_T_su = []
        self.dof_T_su = []
        for i in range(self.n_su):
            dof_T_vdof = TM.from_bounds(bound_dict['su'][:2, :], ['theta', 'd'])
            vdof_T_su = TM.from_bounds(bound_dict['su'][2:, :])
            self.dof_T_vdof.append(dof_T_vdof)
            self.vdof_T_su.append(vdof_T_su)
            self.dof_T_su.append(dof_T_vdof * vdof_T_su)

    def __apply_poses(self, Ts: List[TM], poses: np.ndarray) -> List[TM]:
        assert isinstance(poses, np.ndarray)
        assert len(Ts) == poses.size
        return [T(pose) for T, pose in zip(Ts, poses)]

    def __compute_chains_from_rs(self, dof_T_dof: List[TM]) -> List[TM]:
        assert isinstance(dof_T_dof, list)

        if len(dof_T_dof) == 1:
            return dof_T_dof[0]

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

    def set_poses(self, poses: np.ndarray) -> None:
        assert isinstance(poses, np.ndarray)
        assert poses.size == self.n_joint
        self.poses = poses

    def get_origin_joint_pose(self, i_joint: int, start_joint: int = 0) -> dict:
        T = self.get_origin_joint_TM(i_joint, start_joint)
        return {'position': T.position,
                'orientation': T.q}

    def get_origin_joint_TM(self, i_joint: int, start_joint: int = 0) -> TM:
        """
        The joint number starts from 1 to n in our notation.
        Therefore, i_joint should also start from 1 to n.
        """
        assert 1 <= i_joint <= self.n_joint, \
            print(f'i_joint Should be in between 1 and {self.n_joint}')
        assert start_joint < i_joint, \
            print(f'i_joint={i_joint} should be larger than start_joint {start_joint}')

        if start_joint == 0:
            return self.rs_T_dof[i_joint-1]

        T = self.dof_T_dof[start_joint]
        for i in range(start_joint+1, i_joint):
            T = T * self.dof_T_dof[i]
        return T

    def get_origin_su_pose(self, i_su: int, start_joint: int = 0) -> dict:
        T = self.get_origin_su_TM(i_su, start_joint)
        return {'position': T.position,
                'orientation': T.q}

    def get_origin_su_TM(self, i_su: int, start_joint: int = 0) -> TM:
        """
        The SU number starts from 1 to m in our notation.
        Therefore, i_su should also start from 1 to m.
        """
        assert 1 <= i_su <= self.n_su, \
            print(f'i_su Should be in between 1 and {self.n_su}')

        # Be careful that i_joint starts from 1 to n
        i_joint = self.su_dict[i_su-1]

        assert start_joint <= i_joint, \
            print(f'i_joint {i_joint} which i_su {i_su} is attached to \
                    should be larger than or equal to start_joint {start_joint}')

        if start_joint == 0:
            return self.rs_T_dof[i_joint-1] * self.dof_T_su[i_su-1]

        T = self.dof_T_dof[start_joint]
        for j in range(start_joint+1, i_joint):
            T = T * self.dof_T_dof[j]
        return T * self.dof_T_su[i_su-1]

    def get_current_joint_pose(self, i_joint: int, start_joint: int = 0) -> dict:
        T = self.get_current_joint_TM(i_joint, start_joint)
        return {'position': T.position,
                'orientation': T.q}

    def get_current_su_pose(self, i_su: int, start_joint: int = 0) -> dict:
        T = self.get_current_su_TM(i_su, start_joint)
        return {'position': T.position,
                'orientation': T.q}

    def get_current_joint_TM(self, i_joint: int, start_joint: int = 0) -> TM:
        assert 1 <= i_joint <= self.n_joint, \
            print(f'i_joint Should be in between 1 and {self.n_joint}')
        assert start_joint < i_joint, \
            print(f'i_joint={i_joint} should be larger than start_joint {start_joint}')

        if start_joint == 0:
            return self.rs_Tp_dof[i_joint-1]

        T = self.dof_Tp_dof[start_joint]
        for i in range(start_joint+1, i_joint):
            T = T * self.dof_Tp_dof[i]
        return T

    def get_current_su_TM(self, i_su: int, start_joint: int = 0) -> TM:
        """
        The SU number starts from 1 to m in our notation.
        Therefore, i_su should also start from 1 to m.
        """
        assert 1 <= i_su <= self.n_su, \
            print(f'i_su Should be in between 1 and {self.n_su}')

        # Be careful that i_joint starts from 1 to n
        i_joint = self.su_dict[i_su-1]

        assert start_joint <= i_joint, \
            print(f'i_joint {i_joint} which i_su {i_su} is attached to \
                    should be larger than or equal to start_joint {start_joint}')

        if start_joint == 0:
            return self.rs_Tp_dof[i_joint-1] * self.dof_T_su[i_su-1]

        T = self.dof_Tp_dof[start_joint]
        for j in range(start_joint+1, i_joint):
            T = T * self.dof_Tp_dof[j]
        return T * self.dof_T_su[i_su-1]

    def set_su_dh(self, i_su: int, params: np.ndarray) -> None:
        """
        Sets i_su th SU DH Parameters

        Parameters
        -----------
        i_su: int
            i_su th SU. i_su starts from 1 to m.
        params: np.ndarray
            DH Parameters of the i_su th SU (from its previous DoF)
        """
        assert 1 <= i_su <= self.n_su
        assert params.size == 6
        i = i_su - 1
        self.dof_T_vdof[i] = TM.from_numpy(params[:2, :], ['theta', 'd'])
        self.vdof_T_su[i] = TM.from_numpy(params[2:, :])
        self.dof_T_su.append(self.dof_T_vdof[i] * self.vdof_T_su[i])

    def set_link_dh(self, i_joint: int, params: np.ndarray) -> None:
        assert 1 <= i_joint <= self.n_joint
        assert params.size == 4
        i = i_joint - 1

        self.dof_T_dof[i] = TM.from_numpy(params)
        self.rs_T_dof = self.__compute_chains_from_dof(i, self.dof_T_dof, self.rs_T_dof)
        self.dof_Tp_dof[i] = TM.from_numpy(params)(self.poses[i])
        self.rs_Tp_dof = self.__compute_chains_from_dof(i, self.dof_Tp_dof, self.rs_Tp_dof)
