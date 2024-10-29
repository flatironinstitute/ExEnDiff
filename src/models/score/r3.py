"""Inspired by https://github.com/jasonkyuyim/se3_diffusion/blob/master/data/r3_diffuser.py"""

from math import sqrt
import torch
import math
from src.utils.tensor_utils import inflate_array_like
def adjust_ca_distances(batch_data, target_distance=5):
    """
    Adjust the distances between consecutive Cα atoms in each batch to the target distance.

    Parameters:
    - batch_data: torch.Tensor of shape (N_batch, N_ca, 3) representing the coordinates.
    - target_distance: float, the target distance between consecutive Cα atoms.

    Returns:
    - adjusted_batch_data: torch.Tensor of shape (N_batch, N_ca, 3) with adjusted coordinates.
    """
    N_batch, N_ca, _ = batch_data.shape
    adjusted_batch_data = batch_data.clone()

    for i in range(1, N_ca):
        # Calculate the vector from the previous Cα atom to the current Cα atom
        vector = adjusted_batch_data[:, i, :] - adjusted_batch_data[:, i-1, :]
        # Calculate the current distance
        distance = torch.norm(vector, dim=1, keepdim=True)
        #print(distance)
        # Calculate the scaling factor to adjust the distance to the target distance
        scale = target_distance / distance
        # Scale the vector and adjust the current Cα atom position
        #adjusted_batch_data[:, i, :] = adjusted_batch_data[:, i-1, :] + vector * scale
        adjusted_batch_data[:, i:, :] = adjusted_batch_data[:, i:, :] + vector[:,None,:].repeat(1,adjusted_batch_data[:, i:, :].shape[1],1) * (scale[:,None,:].repeat(1,adjusted_batch_data[:, i:, :].shape[1],adjusted_batch_data[:, i:, :].shape[2]) - 1)
    return adjusted_batch_data
def multivariate_gaussian_log_pdf(x, mean, cov):
    """
    Compute the log pdf of a multivariate Gaussian distribution for each batch.

    Parameters:
    x (torch.Tensor): Input data of shape (N_batch, N_dim).
    mean (torch.Tensor): Mean of the Gaussian distribution of shape (N_batch, N_dim).
    cov (torch.Tensor): Covariance matrix of the Gaussian distribution of shape (N_batch, N_dim, N_dim).

    Returns:
    torch.Tensor: The log pdf values of shape (N_batch,).
    """
    N_batch, N_dim = x.shape

    # Compute the determinant of the covariance matrix
    cov_det = torch.det(cov)  # Shape: (N_batch,)

    # Compute the inverse of the covariance matrix
    cov_inv = torch.inverse(cov)  # Shape: (N_batch, N_dim, N_dim)

    # Compute the normalization constant
    log_normalization_const = -0.5 * (N_dim * torch.log(torch.tensor(2 * torch.pi)) + torch.log(cov_det))  # Shape: (N_batch,)

    # Compute the exponent term
    diff = x - mean  # Shape: (N_batch, N_dim)
    exponent_term = -0.5 * torch.einsum('bi,bij,bj->b', diff, cov_inv, diff)  # Shape: (N_batch,)

    # Compute the log pdf
    log_pdf = log_normalization_const + exponent_term  # Shape: (N_batch,)

    return log_pdf


def compute_mean_distances_per_residue(batch_ca_positions):
    """
    Compute the mean distances between Cα atoms for i to i+1, i to i+3, and i to i+4 for each residue.

    Parameters:
    batch_ca_positions (torch.Tensor): Tensor of shape (N_batch, N_ca, 3) containing Cα positions.

    Returns:
    dict: A dictionary containing the mean distances for 'i_to_i1', 'i_to_i3', and 'i_to_i4' for each residue.
    """
    # Calculate differences for i to i+1
    diff_i1 = batch_ca_positions[:, :-1] - batch_ca_positions[:, 1:]
    dist_i1 = torch.norm(diff_i1, dim=-1)

    # Calculate differences for i to i+3
    diff_i3 = batch_ca_positions[:, :-3] - batch_ca_positions[:, 3:]
    dist_i3 = torch.norm(diff_i3, dim=-1)

    # Calculate differences for i to i+4
    diff_i4 = batch_ca_positions[:, :-4] - batch_ca_positions[:, 4:]
    dist_i4 = torch.norm(diff_i4, dim=-1)

    # Calculate mean distances per residue across all batches
    mean_distances_per_residue = {
        'i_to_i1': dist_i1.mean(dim=0),
        'i_to_i3': dist_i3.mean(dim=0),
        'i_to_i4': dist_i4.mean(dim=0)
    }

    return mean_distances_per_residue
class R3Diffuser:
    """VPSDE diffusion module."""
    def __init__(
        self,
        min_b: float = 0.1,
        max_b: float = 20.0,
        coordinate_scaling: float = 1.0,
    ):
        self.min_b = min_b
        self.max_b = max_b
        self.coordinate_scaling = coordinate_scaling



    def scale(self, x):
        return x * self.coordinate_scaling

    def unscale(self, x):
        return x / self.coordinate_scaling

    def b_t(self, t: torch.Tensor):
        if torch.any(t < 0) or torch.any(t > 1):
            raise ValueError(f'Invalid t={t}')
        return self.min_b + t * (self.max_b - self.min_b)

    def diffusion_coef(self, t):
        return torch.sqrt(self.b_t(t))

    def drift_coef(self, x, t):
        return -0.5 * self.b_t(t) * x

    def sample_prior(self, shape, device=None):
        return torch.randn(size=shape, device=device)

    def marginal_b_t(self, t):
        return t*self.min_b + 0.5*(t**2)*(self.max_b-self.min_b)

    def calc_trans_0(self, score_t, x_t, t):
        beta_t = self.marginal_b_t(t)
        beta_t = beta_t[..., None, None]
        cond_var = 1 - torch.exp(-beta_t)
        return (score_t * cond_var + x_t) / torch.exp(-0.5*beta_t)

    def forward_marginal(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor
    ):
        """Samples marginal p(x(t) | x(0)).

        Args:
            x_0: [..., n, 3] initial positions in Angstroms.
            t: continuous time in [0, 1].

        Returns:
            x_t: [..., n, 3] positions at time t in Angstroms.
            score_t: [..., n, 3] score at time t in scaled Angstroms.
        """
        t = inflate_array_like(t, x_0)
        x_0 = self.scale(x_0)

        loc = torch.exp(-0.5 * self.marginal_b_t(t)) * x_0
        scale = torch.sqrt(1 - torch.exp(-self.marginal_b_t(t)))
        z = torch.randn_like(x_0)
        x_t = z * scale + loc
        score_t = self.score(x_t, x_0, t)

        x_t = self.unscale(x_t)
        return x_t, score_t

    def score_scaling(self, t: torch.Tensor):
        return 1.0 / torch.sqrt(self.conditional_var(t))

    def reverse(
        self,
        x_t: torch.Tensor,
        score_t: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        mask: torch.Tensor = None,
        center: bool = True,
        noise_scale: float = 1.0,
        probability_flow: bool = True,
        conditional_operator=None,
        conditional_noise=0,
        conditional_multi_noise = 0,
        N = 1,
        T = 1,
        ts = None,
        samples = None,
    ):
        """Simulates the reverse SDE for 1 step

        Args:
            x_t: [..., 3] current positions at time t in angstroms.
            score_t: [..., 3] rotation score at time t.
            t: continuous time in [0, 1].
            dt: continuous step size in [0, 1].
            mask: True indicates which residues to diffuse.
            probability_flow: whether to use probability flow ODE.

        Returns:
            [..., 3] positions at next step t-1.
        """
        # conditioning
        discrete_betas = self.b_t(torch.from_numpy(ts[::-1].copy())) / N
        discrete_betas.requires_grad = True
        alphas = 1. - discrete_betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        with torch.enable_grad():
            x_t = self.scale(x_t)
            #print(x_t.requires_grad)
            timestep = (t * (N - 1) / T).long()
            alpha_t = alphas_cumprod.to(x_t.device)[timestep]
            one_minus_alpha_t = 1 - alpha_t

            x0 = (x_t + one_minus_alpha_t[:, None, None] * score_t) / torch.sqrt(alpha_t[:, None, None])
            if center:
                mask = torch.ones_like(x_t[..., 0])
                com = torch.sum(x0, dim=-2) / torch.sum(mask, dim=-1)[..., None]  # reduce length dim
                x0 -= com[..., None, :]
            x0_unscale = self.unscale(x0)

            y_value_list = [None] * len(conditional_operator)
            A_value_list = [None] * len(conditional_operator)
            len_list = [None] * (len(conditional_operator) + 1)
            len_list[0] = 0
            for num_i in range(len(conditional_operator)):
                y_value_list[num_i] = torch.from_numpy(samples[num_i]).to(device=x_t.device,
                                                                          dtype=torch.float32)
                A_value_list[num_i] = conditional_operator[num_i](x0_unscale)
                len_list[num_i + 1] = len_list[num_i] + y_value_list[num_i].shape[-1]
            if len(conditional_operator) == 1:
                y_value = y_value_list[0]
                A_x0 = A_value_list[0]
            else:
                y_value = torch.cat(y_value_list, dim=1).to(device=x_t.device, dtype=torch.float32)
                A_x0 = torch.cat(A_value_list, dim=1).to(device=x_t.device, dtype=torch.float32)
            y_value.requires_grad = True
            norm = (A_x0 - y_value).pow(2)
            weight = torch.ones(norm.shape[1]).to(x_t.device)
            for num_op in range(len(conditional_operator)):
                weight[int(len_list[num_op]):int(len_list[num_op + 1])] = torch.ones(
                    int(len_list[num_op + 1]) - int(len_list[num_op])).to(x_t.device) * conditional_multi_noise[
                                                                              num_op]
            norm = norm * weight
            if len(norm.shape) == 1:
                norm_sqrt = norm.sqrt()
            elif len(norm.shape) == 2:
                norm_sqrt = torch.sum(norm, dim=-1).sqrt()

            norm_sqrt = torch.where(norm_sqrt == 0, torch.tensor(1.0, device=norm_sqrt.device), norm_sqrt)
            norm_final = norm.sum()
            grad_log_p = torch.autograd.grad(norm_final, x_t, create_graph=True, retain_graph=True)[0]
            grad_log_p = grad_log_p.detach()
            correct_term = 1 * conditional_noise[0] / norm_sqrt[:, None, None] * grad_log_p
            score_t -= correct_term

            t = inflate_array_like(t, x_t)
            f_t = self.drift_coef(x_t, t)
            g_t = self.diffusion_coef(t)

            z = noise_scale * torch.randn_like(score_t)

            rev_drift = (f_t - g_t ** 2 * score_t) * dt * (0.5 if probability_flow else 1.)
            rev_diffusion = 0. if probability_flow else (g_t * sqrt(dt) * z)
            perturb = rev_drift + rev_diffusion

            if mask is not None:
                perturb *= mask[..., None]
            else:
                mask = torch.ones_like(x_t[..., 0])
            x_t_1 = x_t - perturb  # reverse in time
            if center:
                com = torch.sum(x_t_1, dim=-2) / torch.sum(mask, dim=-1)[..., None]  # reduce length dim
                x_t_1 -= com[..., None, :]

            x_t_1 = self.unscale(x_t_1)
            return x_t_1.detach()

    def conditional_var(self, t, use_torch=False):
        """Conditional variance of p(xt|x0).
        Var[x_t|x_0] = conditional_var(t) * I
        """
        return 1.0 - torch.exp(-self.marginal_b_t(t))

    def score(self, x_t, x_0, t, scale=False):
        t = inflate_array_like(t, x_t)
        if scale:
            x_t, x_0 = self.scale(x_t), self.scale(x_0)
        return -(x_t - torch.exp(-0.5 * self.marginal_b_t(t)) * x_0) / self.conditional_var(t)

    def distribution(self, x_t, score_t, t, mask, dt):
        x_t = self.scale(x_t)
        f_t = self.drift_coef(x_t, t)
        g_t = self.diffusion_coef(t)
        std = g_t * sqrt(dt)
        mu = x_t - (f_t - g_t**2 * score_t) * dt
        if mask is not None:
            mu *= mask[..., None]
        return mu, std