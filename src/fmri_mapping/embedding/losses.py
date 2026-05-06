import torch
from torch import nn
import torch.nn.functional as F


def geometry_cosine_reg(z_in, z_out):
    zi = F.normalize(z_in, dim=1)
    zo = F.normalize(z_out, dim=1)
    Si = zi @ zi.T
    So = zo @ zo.T
    return F.mse_loss(So, Si)


def info_nce_symmetric(
    z_a: torch.Tensor, z_b: torch.Tensor, temperature: float = 0.07
) -> torch.Tensor:
    """
    z_a, z_b: (B, D)
    Uses in-batch negatives. Assumes row i in a matches row i in b.
    """
    z_a = F.normalize(z_a, dim=1)
    z_b = F.normalize(z_b, dim=1)

    logits = (z_a @ z_b.T) / temperature  # (B,B)
    targets = torch.arange(logits.size(0), device=logits.device)

    loss_ab = F.cross_entropy(logits, targets)
    loss_ba = F.cross_entropy(logits.T, targets)
    return 0.5 * (loss_ab + loss_ba)


def pull_cosine(a, b):
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)

    return (1.0 - (a * b).sum(dim=1)).mean()


def circle_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    *,
    margin: float = 0.25,
    gamma: float = 32.0,
):
    """
    Circle Loss for paired embeddings with in-batch negatives.

    Args
    ----
    z_a, z_b : (B, D)
        Paired embeddings (e.g., two repeats of the same image).
    margin : float
        Margin parameter (m in the paper). Typical: 0.25
    gamma : float
        Scale factor. Typical: 32-80 (lower for noisy embeddings).

    Returns
    -------
    loss : scalar tensor
    """

    # Normalize
    z_a = F.normalize(z_a, dim=1)
    z_b = F.normalize(z_b, dim=1)

    # Similarity matrix
    sim = z_a @ z_b.T  # (B, B)

    B = sim.size(0)
    device = sim.device

    # Positive similarities: diagonal
    sp = sim.diag()  # (B,)

    # Negative similarities: off-diagonal
    mask = torch.eye(B, device=device, dtype=torch.bool)
    sn = sim[~mask].view(B, B - 1)  # (B, B-1)

    # Circle loss weighting
    alpha_p = torch.clamp_min(1 + margin - sp.detach(), 0.0)
    alpha_n = torch.clamp_min(sn.detach() + margin, 0.0)

    delta_p = 1 - margin
    delta_n = margin

    # Logits
    logit_p = -gamma * alpha_p * (sp - delta_p)  # (B,)
    logit_n = gamma * alpha_n * (sn - delta_n)  # (B,B-1)

    # Final loss
    loss_p = torch.logsumexp(logit_p, dim=0)
    loss_n = torch.logsumexp(logit_n, dim=1)

    loss = F.softplus(loss_p + loss_n).mean()
    return loss


def vsp_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    add_cycle_term: bool = False,
    eps: float = 1e-10,
) -> torch.Tensor:
    """View Similarity Preservation (VSP) loss between two embedding sets.

    This loss enforces that the pairwise cosine similarity structure
    among translated embeddings `y` matches that of the original
    embeddings `x`.

    Args:
        x (torch.Tensor): Original embeddings, shape (N, D).
        y (torch.Tensor): Translated embeddings, shape (N, D).
        add_cycle_term (bool, optional): Whether to include the cycle
            consistency term in the loss. Defaults to False.
        eps (float, optional): Small constant for numerical stability.

    Returns:
        torch.Tensor: Scalar loss value.
    """
    # Normalize to unit length
    x_norm = x / (x.norm(dim=1, keepdim=True) + eps)
    y_norm = y / (y.norm(dim=1, keepdim=True) + eps)

    # Pairwise cosine similarities
    sim_xx = x_norm @ x_norm.T  # true geometry
    sim_yy = y_norm @ y_norm.T  # predicted geometry
    sim_xy = y_norm @ x_norm.T  # cross geometry

    # Geometry preservation and reflection losses
    loss_geom = (sim_xx - sim_yy).abs().mean()
    loss_reflect = (sim_xx - sim_xy).abs().mean()
    loss = loss_geom + loss_reflect

    if add_cycle_term:
        loss = loss + (sim_yy - sim_xy).abs().mean()

    return loss


def loo_repeat_denoise_loss(z1, z2, z3):
    """
    Leave-one-out repeat denoising.

    Encourages each embedding to move toward
    the mean of the other repeats.
    """

    t1 = 0.5 * (z2 + z3)
    t2 = 0.5 * (z1 + z3)
    t3 = 0.5 * (z1 + z2)

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    z3 = F.normalize(z3, dim=1)

    t1 = F.normalize(t1, dim=1)
    t2 = F.normalize(t2, dim=1)
    t3 = F.normalize(t3, dim=1)

    loss = 1 - (z1 * t1).sum(dim=1) + 1 - (z2 * t2).sum(dim=1) + 1 - (z3 * t3).sum(dim=1)

    return loss.mean()


def coral_loss(Cx, Cy, eps=1e-5):
    # Cx, Cy: (B,D)
    mx, my = Cx.mean(0), Cy.mean(0)
    Cx0, Cy0 = Cx - mx, Cy - my
    covx = (Cx0.T @ Cx0) / (Cx.size(0) - 1 + eps)
    covy = (Cy0.T @ Cy0) / (Cy.size(0) - 1 + eps)
    return F.mse_loss(mx, my) + F.mse_loss(covx, covy)


def vicreg_terms(C, var_target=1.0, eps=1e-4):
    # variance term
    std = torch.sqrt(C.var(dim=0) + eps)
    var_loss = torch.mean(F.relu(var_target - std))
    # covariance term
    C0 = C - C.mean(0)
    cov = (C0.T @ C0) / (C.size(0) - 1 + eps)
    off = cov - torch.diag(torch.diag(cov))
    cov_loss = (off**2).mean()
    return var_loss, cov_loss


def orthogonality_loss_linear(layer: nn.Linear) -> torch.Tensor:
    """
    Orthogonality regularizer for a linear layer (expects weight shape [out, in]).
    Typically used when out == in and bias == False.
    """
    W = layer.weight
    I = torch.eye(W.size(0), device=W.device, dtype=W.dtype)
    return ((W.T @ W - I) ** 2).mean()


