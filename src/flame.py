import torch
import torch.nn as nn
import numpy as np
from numpy.typing import NDArray
import pickle
import trimesh
import tyro
from typing import Optional, Tuple

from src.utils import rodrigues_rotation, make_homogeneous, blendshape


class FLAME(nn.Module):
    """Differentiable Head Parametric Model - FLAME"""
    TOTAL_SHAPES = 300
    TOTAL_EXPRESSIONS = 100
    TOTAL_JOINTS = 5
    TOTAL_VERTICES = 5023
    TOTAL_FACES = 9976
    JOINT_DESC = {
        0: "Head",
        1: "Neck",
        2: "Jaw",
        3: "Left eye",
        4: "Right eye"
    }

    def __init__(
        self,
        checkpoint_path: str = "data/generic_flame2023.pkl",
        num_shapes: int = TOTAL_SHAPES,
        num_expressions: int = TOTAL_EXPRESSIONS,
        control_eyelids: bool = False,
        dtype: torch.dtype = torch.float64,
        device: str = "cpu"
    ):
        super(FLAME, self).__init__()
        self.control_eyelids = control_eyelids
        if self.control_eyelids:
            self.TOTAL_EXPRESSIONS += 2
        assert num_shapes <= self.TOTAL_SHAPES
        assert num_expressions <= self.TOTAL_EXPRESSIONS
        self.num_shapes = num_shapes
        self.num_expressions = num_expressions
        self.num_joints = self.TOTAL_JOINTS
        self.dtype = dtype
        self.device = device
        # load the model weights
        with open(checkpoint_path, 'rb') as f:
            weights = pickle.load(f, encoding='latin1')
        # initialize the fixed parameters
        self.shapes = torch.zeros((1, self.num_shapes), dtype=self.dtype, device=self.device, requires_grad=False)
        self.expressions = torch.zeros((1, self.num_expressions), dtype=self.dtype, device=self.device, requires_grad=False)
        # initialize the tail unused parameters
        self.shapes_tail = torch.zeros((1, self.TOTAL_SHAPES - self.num_shapes), dtype=self.dtype, device=self.device, requires_grad=False)
        self.expressions_tail = torch.zeros((1, self.TOTAL_EXPRESSIONS - self.num_expressions), dtype=self.dtype, device=self.device, requires_grad=False)
        # initialize the rotation parameters
        self.pose = torch.zeros((1, 3), dtype=self.dtype, device=self.device, requires_grad=False)
        self.neck = torch.zeros((1, 3), dtype=self.dtype, device=self.device, requires_grad=False)
        self.jaw = torch.zeros((1, 3), dtype=self.dtype, device=self.device, requires_grad=False)
        self.left_eye = torch.zeros((1, 3), dtype=self.dtype, device=self.device, requires_grad=False)
        self.right_eye = torch.zeros((1, 3), dtype=self.dtype, device=self.device, requires_grad=False)
        # extract the parameters from the weights
        self.faces_numpy = np.array(weights["f"]).astype(np.int64)
        self.register_buffer("faces", torch.tensor(self.faces_numpy, dtype=torch.int64, device=self.device))
        # joints regressor from vertices weights
        self.register_buffer(
            "joints_weights", torch.tensor(np.array(weights["J_regressor"].todense()), dtype=self.dtype, device=self.device)
        )
        # the linear blend skinning transforms weights
        self.register_buffer(
            "lbs_weights", torch.tensor(np.array(weights["weights"]), dtype=self.dtype, device=self.device)
        )
        # vertices static template
        self.register_buffer(
            "template", torch.tensor(np.array(weights["v_template"]), dtype=self.dtype, device=self.device)
        )
        # pose corrective directions basis
        self.register_buffer(
            "posedirs",
            torch.tensor(
                np.array(weights["posedirs"]).reshape(-1, weights["posedirs"].shape[-1]).transpose(),
                dtype=self.dtype,
                device=self.device
            )
        )
        if self.control_eyelids:
            # control eyelids with additional expression blendshapes
            left_eyelid = np.load("data/l_eyelid.npy")
            right_eyelid = np.load("data/r_eyelid.npy")
            weights["shapedirs"] = np.concatenate(
                (
                    weights["shapedirs"][:, :, :300],
                    left_eyelid[:, :, None],
                    right_eyelid[:, :, None],
                    weights["shapedirs"][:, :, 300:]
                ),
                axis=2
            )
        # shape directions basis
        self.register_buffer(
            "shapedirs",
            torch.tensor(
                np.array(weights["shapedirs"]).reshape(-1, weights["shapedirs"].shape[-1]).transpose(),
                dtype=self.dtype,
                device=self.device
            )
        )
        # indices of parents in kinematic tree
        parents = np.array(weights["kintree_table"][0])
        parents[0] = -1
        self.register_buffer(
            "parents", torch.tensor(parents, dtype=torch.int64, device=self.device)
        )
        # identity matrix for further computations
        self.__identity = torch.eye(3, dtype=self.dtype, device=self.device).unsqueeze(0)

    def forward(
        self,
        shape: Optional[torch.Tensor] = None,
        expression: Optional[torch.Tensor] = None,
        pose: Optional[torch.Tensor] = None,
        neck: Optional[torch.Tensor] = None,
        jaw: Optional[torch.Tensor] = None,
        left_eye: Optional[torch.Tensor] = None,
        right_eye: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes FLAME vertices given the control parameters
        Parameters
        ----------
            shape: (N, S) shape coefficients
            expression: (N, E) expression coefficients
            pose: (N, 3) global rigid head rotation
            neck: (N, 3) neck rotation (relative to the head rest pose)
            jaw: (N, 3) jaw rotation (relative to the neck rest pose)
            left_eye: (N, 3) left eye rotation (relative to the neck rest pose)
            right_eye: (N, 3) right eye rotation (relative to the neck rest pose)
        Returns
        -------
            vertices: N X V X 3
            relative_joints: N x J X 3
        """

        shape = shape if shape is not None else self.shapes
        expression = expression if expression is not None else self.expressions
        pose = pose if pose is not None else self.pose
        neck = neck if neck is not None else self.neck
        jaw = jaw if jaw is not None else self.jaw
        left_eye = left_eye if left_eye is not None else self.left_eye
        right_eye = right_eye if right_eye is not None else self.right_eye

        coeffs = torch.cat([shape, self.shapes_tail, expression, self.expressions_tail], dim=1)
        full_pose = torch.cat([pose, neck, jaw, left_eye, right_eye], dim=1)

        batch_size = coeffs.shape[0]

        # apply shape and expression components
        shaped = blendshape(self.template.unsqueeze(0), coeffs, self.shapedirs)  # (N, V, 3)

        # apply pose changes
        rotations = rodrigues_rotation(full_pose.reshape(-1, 3)).reshape(batch_size, -1, 3, 3)  # (N, P, 3, 3)
        pose_features = (rotations[:, 1:, :, :] - self.__identity).reshape(batch_size, -1)  # (N, (P-1) * 3 * 3)
        posed = blendshape(shaped, pose_features, self.posedirs)  # (N, V, 3)

        # regress joints positions
        joints = self.joints_weights @ shaped  # (J, V) @ (N, V, 3) -> (N, J, 3)
        # compute relative joints positions w.r.t its parents
        relative_joints = joints.clone()
        relative_joints[:, 1:] -= joints[:, self.parents[1:]]

        # compute transformation matrices (N, J, 4, 4)
        homogeneous_4d = make_homogeneous(rotations, joints)
        translation_4d = make_homogeneous(
            rotations=self.__identity.unsqueeze(0).repeat(batch_size, joints.size(1), 1, 1),
            translations=-joints
        )

        # compute the accumulated transformations w.r.t root joint for each joint
        transforms = [homogeneous_4d[:, 0] @ translation_4d[:, 0]]
        for i in range(1, self.parents.shape[0]):
            transform = transforms[self.parents[i]] @ homogeneous_4d[:, i] @ translation_4d[:, i]
            transforms.append(transform)
        transforms = torch.stack(transforms, dim=1)

        # linear blend skinning - compute weighted transformations and apply it
        weighted_transforms = torch.matmul(
            self.lbs_weights.repeat(batch_size, 1, 1),
            transforms.reshape(batch_size, -1, 16)
        ).reshape(batch_size, -1, 4, 4)  # (N, V, J) @ (N, J, 16) -> (N, V, 16) -> (N, V, 4, 4)
        vertices = weighted_transforms[:, :, :3, :3] @ posed[:, :, :, None] + weighted_transforms[:, :, :3, 3:4]  # (N, V, 3, 3) @ (N, V, 3, 1) -> (N, V, 3, 1)
        vertices = vertices[..., 0]  # (N, V, 3)
        return vertices, relative_joints

    def run(
        self,
        shape: Optional[NDArray] = None,
        expression: Optional[NDArray] = None,
        pose: Optional[NDArray] = None,
        neck: Optional[NDArray] = None,
        jaw: Optional[NDArray] = None,
        left_eye: Optional[NDArray] = None,
        right_eye: Optional[NDArray] = None,
    ):
        """
        Computes FLAME vertices given the control parameters
        Parameters
        ----------
            shape: (S,) shape coefficients
            expression: (E,) expression coefficients
            pose: (3,) global rotation of the neck+head
            neck: (3,) head rotation (relative to the neck rest pose)
            jaw: (3,) jaw rotation (relative to the head rest pose)
            left_eye: (3,) left eye rotation (relative to the head rest pose)
            right_eye: (3,) right eye rotation (relative to the head rest pose)
        Returns
        -------
            vertices: V X 3
            relative_joints: J X 3
        """

        shape = torch.tensor(shape).unsqueeze(0) if shape is not None else self.shapes
        expression = torch.tensor(expression).unsqueeze(0) if expression is not None else self.expressions
        pose = torch.tensor(pose).unsqueeze(0) if pose is not None else self.pose
        neck = torch.tensor(neck).unsqueeze(0) if neck is not None else self.neck
        jaw = torch.tensor(jaw).unsqueeze(0) if jaw is not None else self.jaw
        left_eye = torch.tensor(left_eye).unsqueeze(0) if left_eye is not None else self.left_eye
        right_eye = torch.tensor(right_eye).unsqueeze(0) if right_eye is not None else self.right_eye

        with torch.no_grad():
            vertices, relative_joints = self.forward(
                shape=shape,
                expression=expression,
                pose=pose,
                neck=neck,
                jaw=jaw,
                left_eye=left_eye,
                right_eye=right_eye
            )

        vertices = vertices[0].cpu().numpy()  # type: ignore
        relative_joints = relative_joints[0].cpu().numpy()  # type: ignore
        return vertices, relative_joints


def main(
    checkpoint_path: str = 'data/generic_flame2023.pkl',
    save: bool = False,
    seed: int = 0,
    control_eyelids: bool = False
) -> None:
    """Generate a random FLAME mesh and optionally save it."""
    np.random.seed(seed)

    model = FLAME(checkpoint_path=checkpoint_path, control_eyelids=control_eyelids)

    # Generate random control parameters within realistic range
    shape = np.random.normal(size=(model.num_shapes,))
    expression = np.random.normal(size=(model.num_expressions,))
    pose = np.pi * np.random.uniform(size=(3,)) / 10
    neck = np.pi * np.random.uniform(size=(3,)) / 10
    jaw = np.pi * np.random.uniform(size=(3,)) / 10
    left_eye = np.pi * np.random.uniform(size=(3,)) / 10
    right_eye = np.pi * np.random.uniform(size=(3,)) / 10

    vertices, _ = model.run(
        shape=shape,
        expression=expression,
        pose=pose,
        neck=neck,
        jaw=jaw,
        left_eye=left_eye,
        right_eye=right_eye
    )

    if save:
        # save vertices as .obj mesh and control parameters as .npz
        mesh = trimesh.Trimesh(vertices, model.faces_numpy)
        mesh.export('flame_output.obj')
        np.savez('flame_params.npz', shape=shape, expression=expression, pose=pose, neck=neck, jaw=jaw, left_eye=left_eye, right_eye=right_eye)


if __name__ == '__main__':
    tyro.cli(main, description="Generate random FLAME mesh")
