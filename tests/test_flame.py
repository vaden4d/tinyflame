import trimesh
import numpy as np
from src.flame import FLAME


def test_flame(eps: float = 1e-8) -> None:
    # load inputs and outputs
    params = np.load('tests/data/flame_params.npz')
    mesh = trimesh.load('tests/data/flame_output.obj')

    model = FLAME()
    vertices, _ = model.run(**params)

    error = np.abs(mesh.vertices - vertices).max()  # type: ignore
    assert error <= eps, f"The difference between the vertices is {error}"
