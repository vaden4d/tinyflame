import torch


def blendshape(
    template: torch.Tensor,
    coeffs: torch.Tensor,
    shapes: torch.Tensor
) -> torch.Tensor:
    """
    Blends the basis shapes with the coefficients and adds to the template
    Parameters
    ----------
        templates: N x V x 3 or 1 x V x 3, mean template meshes
        coeffs: N x S, blendshape coefficients
        dirs: S x V * 3, blendshape shapes (directions)
    Returns
    -------
        vertices: N x V x 3, mesh vertices
    """
    offsets = coeffs @ shapes
    vertices = template + offsets.reshape(-1, template.size(1), 3)
    return vertices


def skew_symmetric(
    vectors: torch.Tensor
) -> torch.Tensor:
    """
    Computes the skew-symmetric matrices per vector
    Parameters
    ----------
        vectors: N x 3, input vectors
    Returns
    -------
        skew: N x 3 x 3, skew-symmetric matrices
    """
    identity = torch.eye(3, dtype=vectors.dtype, device=vectors.device).unsqueeze(0)
    skew = torch.cross(identity.expand(vectors.size(0), -1, -1), vectors.unsqueeze(1), dim=2)
    return skew


def rodrigues_rotation(
    vectors: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Calculates the rotation matrices around axes using Rodrigues formula.
    Parameters
    ----------
        vectors: N x 3, axes directions
        eps: float, small value to avoid division by zero
    Returns
    -------
        rotations: N x 3 x 3, rotation matrices
    """
    device, dtype = vectors.device, vectors.dtype
    # length equals to rotation angles
    angles = torch.norm(vectors, dim=1, keepdim=True) + eps
    dirs = vectors / angles
    cos = torch.cos(angles).unsqueeze(1)
    sin = torch.sin(angles).unsqueeze(1)
    # setup skew-symmetric matrices correspond to the rotation dirs
    identity = torch.eye(3, dtype=dtype, device=device).unsqueeze(0)
    skew = skew_symmetric(dirs)
    rotations = identity + sin * skew + (1 - cos) * torch.bmm(skew, skew)
    return rotations


def make_homogeneous(
    rotations: torch.Tensor,
    translations: torch.Tensor
) -> torch.Tensor:
    """
    Creates a homogeneous matrix from rotations and translations
    Parameters
    ----------
        rotations: N x 3 x 3 or N x J x 3 x 3, rotation matrices
        translations: N x 3 or N x J x 3, translation vectors
    Returns
    -------
        matrices: N x 4 x 4 or N x J x 4 x 4, homogeneous matrices
    """
    is_3d = rotations.ndim == 3 and translations.ndim == 2
    is_4d = rotations.ndim == 4 and translations.ndim == 3
    assert is_3d or is_4d, "Invalid input shapes"

    if is_4d:
        batch_size, num_points, _, _ = rotations.size()
        rotations = rotations.reshape(-1, 3, 3)
        translations = translations.reshape(-1, 3)

    matrix = torch.zeros((rotations.shape[0], 4, 4), dtype=rotations.dtype, device=rotations.device)
    matrix[:, :3, :3] = rotations
    matrix[:, :3, 3] = translations
    matrix[:, 3, 3] = 1.0

    if is_4d:
        matrix = matrix.reshape(batch_size, num_points, 4, 4)
    return matrix
