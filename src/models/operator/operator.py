import os
from typing import Any, Dict, Tuple, Optional
from random import random
from copy import deepcopy

import numpy as np
import torch
from lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric

from src.models.score.frame import FrameDiffuser
from src.models.loss import ScoreMatchingLoss
from src.common.rigid_utils import Rigid
from src.common.all_atom import compute_backbone
from src.common.pdb_utils import atom37_to_pdb, merge_pdbfiles
import omegaconf
import mdtraj as md
import pickle
from src.models.score import so3, r3
from src.common.rigid_utils import Rigid, Rotation, quat_multiply
from src.common import rotation3d
import src.models.operator.operator_util as outil
import src.models.operator.cryoEM_operator_util as cutil
import src.models.operator.dssp_util as dutil

_OPERATORS = {}
def get_operator(name):
    return _OPERATORS[name]

def register_operator(cls=None, *, name=None):
    """A decorator for registering operator classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _OPERATORS:
            raise ValueError(f'Already registered model with name: {local_name}')
        _OPERATORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)

@register_operator(name="eed_operator")
def test_eed_operator(coords):
    """
    Compute the end to end distance for a batch of protein structures.
    """
    first_atom = coords[:, 0, :]
    last_atom = coords[:, -1, :]
    squared_diff = (first_atom - last_atom) ** 2
    squared_distance = torch.sum(squared_diff, dim=1)
    distance = torch.sqrt(squared_distance)
    distance = distance.view(-1, 1)
    return distance

@register_operator(name="rg_operator")
def radius_of_gyration(coords):
    """
    Compute the radius of gyration for a batch of protein structures.
    """
    centered_coords = coords - coords.mean(dim=1, keepdim=True)
    sq_distances = torch.sum(centered_coords ** 2, dim=2)
    rg = torch.sqrt(torch.mean(sq_distances, dim=1))
    rg = rg.view(-1, 1)
    return rg

@register_operator(name="rmsd_operator")
def calculate_rmsd_batch(pos, protein_name="chignolin"):
    """
    Calculate the root mean square deviation (RMSD) between two sets of points pos and ref for batches.

    Args:
    pos (torch.Tensor): A tensor of shape (B, N, 3) representing the positions of the first set of points in batches.
    ref (torch.Tensor): A tensor of shape (B, N, 3) representing the positions of the second set of points in batches.

    Returns:
    torch.Tensor: RMSD between the two sets of points for each batch.
    """
    ref = outil.read_CA_pos(protein_name=protein_name, N_batch=pos.shape[0]).double() * 10 #convert nm to Angstrom
    ref = ref.to(pos.device)
    if pos.shape[1] != ref.shape[1]:
        raise ValueError("pos and ref must have the same number of points")
    R, t = find_alignment_kabsch_batch(ref, pos)
    ref0 = torch.matmul(R, ref.transpose(1, 2)).transpose(1, 2) + t.unsqueeze(1)
    rmsd = torch.linalg.norm(ref0 - pos, dim=2).mean(dim=1).view(-1, 1)
    return rmsd

@register_operator(name="dssp_operator")
def continuous_assign_dssp(ca_positions):
    # print("coord shape",coord.shape)
    coord = dutil.recover_backbone_atoms(ca_positions)
    # check input
    # coord, org_shape = _check_input(coord)
    # get hydrogen bond map
    hbmap = dutil.get_hbond_map(coord)
    hbmap = dutil.rearrange(hbmap, '... l1 l2 -> ... l2 l1')  # convert into "i:C=O, j:N-H" form
    # Extract diagonals with offsets 3, 4, 5
    turn3 = torch.diagonal(hbmap, dim1=-2, dim2=-1, offset=3)
    turn4 = torch.diagonal(hbmap, dim1=-2, dim2=-1, offset=4)
    turn5 = torch.diagonal(hbmap, dim1=-2, dim2=-1, offset=5)

    # assignment of helical sses
    h3 = torch.nn.functional.pad(turn3[:, :-1] * turn3[:, 1:], [1, 3])
    h4 = torch.nn.functional.pad(turn4[:, :-1] * turn4[:, 1:], [1, 4])
    h5 = torch.nn.functional.pad(turn5[:, :-1] * turn5[:, 1:], [1, 5])
    helix4 = h4 + torch.roll(h4, 1, 1) + torch.roll(h4, 2, 1) + torch.roll(h4, 3, 1)
    h3 = h3 * (1 - torch.roll(helix4, -1, 1)) * (1 - helix4)  # helix4 is higher prioritized
    h5 = h5 * (1 - torch.roll(helix4, -1, 1)) * (1 - helix4)  # helix4 is higher prioritized
    helix3 = h3 + torch.roll(h3, 1, 1) + torch.roll(h3, 2, 1)
    helix5 = h5 + torch.roll(h5, 1, 1) + torch.roll(h5, 2, 1) + torch.roll(h5, 3, 1) + torch.roll(h5, 4, 1)
    helix = helix3 + helix4 + helix5

    # identify bridge
    unfoldmap = hbmap.unfold(-2, 3, 1).unfold(-2, 3, 1).contiguous()
    unfoldmap_rev = unfoldmap.transpose(-4, -3).contiguous()
    p_bridge = (unfoldmap[:, :, :, 0, 1] * unfoldmap_rev[:, :, :, 1, 2]) + (
                unfoldmap_rev[:, :, :, 0, 1] * unfoldmap[:, :, :, 1, 2])
    p_bridge_2 = torch.nn.functional.pad(p_bridge, [1, 1, 1, 1])
    a_bridge = (unfoldmap[:, :, :, 1, 1] * unfoldmap_rev[:, :, :, 1, 1]) + (
                unfoldmap[:, :, :, 0, 2] * unfoldmap_rev[:, :, :, 0, 2])
    a_bridge_2 = torch.nn.functional.pad(a_bridge, [1, 1, 1, 1])
    # ladder
    ladder = (p_bridge_2 + a_bridge_2).sum(-1)
    return torch.cat([helix, ladder], dim=-1)

@register_operator(name="helix_dist_operator")
def helix_dist_operator(batch_ca_positions):
    diff_i3 = batch_ca_positions[:, :-3] - batch_ca_positions[:, 3:]
    dist_i3 = torch.norm(diff_i3, dim=-1)
    diff_i4 = batch_ca_positions[:, :-4] - batch_ca_positions[:, 4:]
    dist_i4 = torch.norm(diff_i4, dim=-1)
    return torch.cat([dist_i3, dist_i4], dim=1)

@register_operator(name="helix_rmsd_operator")
def real_helix_operator(batch_ca_positions):
    ref_helix = np.load("/export/users/liu3307/Str2Str/src/models/rmsd_analysis/helix_nmr.npy")
    ref_helix = torch.from_numpy(ref_helix).to(batch_ca_positions.device) * 10
    ref_helix = ref_helix.unsqueeze(0).repeat(batch_ca_positions.shape[0], 1, 1).double()
    rmsd = outil.calculate_rmsd_batch(batch_ca_positions, ref_helix)
    return rmsd

@register_operator(name="beta_rmsd_operator")
def real_beta_operator(A, B, parallel_structure_file,anti_parallel_structure_file):
    # A:indice
    # B:ca_pos
    B = B.float()
    N_batch, N_dim, N_features = B.shape
    ref_beta_parallel = np.load(parallel_structure_file)
    ref_beta_parallel = torch.from_numpy(ref_beta_parallel).to(B.device).float() * 10
    random_index = torch.randint(0, ref_beta_parallel.shape[0], (1,)).item()
    ref_beta_parallel = ref_beta_parallel[random_index]

    ref_beta_antiparallel = np.load(anti_parallel_structure_file)
    ref_beta_antiparallel = torch.from_numpy(ref_beta_antiparallel).to(B.device) * 10
    random_index = torch.randint(0, ref_beta_antiparallel.shape[0], (1,)).item()
    ref_beta_antiparallel = ref_beta_antiparallel[random_index]

    beta_pair_mask, beta_indice, beta_pair_indice = outil.get_beta_indice(A)
    beta_pos = outil.get_beta_position(B, beta_indice)
    beta_output_parallel = outil.beta_func1(beta_pos, beta_pair_mask, beta_pair_indice, N_dim, ref_beta_parallel)
    beta_output_antiparallel = outil.beta_func1(beta_pos, beta_pair_mask, beta_pair_indice, N_dim, ref_beta_antiparallel)
    return beta_output_parallel + beta_output_antiparallel

@register_operator(name="cryoem_operator")
def generate_images(
        coord,
        n_pixel=32,  ## use power of 2 for CTF purpose
        pixel_size=1,
        sigma=1.0,
        snr=np.infty,
        rotation=False,
        add_ctf=False,
        defocus_min=0.027,
        defocus_max=0.090,
        protein_name="bba",
):
    ##
    ## generate_images : function : generate synthetic cryo-EM images, at random orientation (and random CTF), given a set of structures
    ##
    ## Input :
    ##     coord : numpy ndarray or torch tensor of float of shape (N_image, N_atom, 3) : 3D Cartesian coordinates of atoms of configuration aligned to generate the synthetic images
    ##     n_pixel : int : number of pixels of image i.e. the synthetic (cryo-EM) images are of shape (n_pixel, n_pixel)
    ##     pixel_size : float : width of each pixel in physical space in Angstrom
    ##     sigma : float : Gaussian width of each atom in the imaging model in Angstrom
    ##     snr : float : Signal-to-noise (SNR) for adding noise to the image, if snr = np.infty, does not add noise to the images
    ##     add_ctf : bool : If True, add Contrast transfer function (CTF) to the synthetic images.
    ##     batch_size : int : to split the set of images into batches for calculation, where structure in the same batch are fed into calculation at the same time, a parameter for computational performance / memory management
    ##     device : str : "cuda" or "cpu", to be fed into pyTorch, see pyTorch manual for more detail
    ## Output :
    ##     rot_mats : torch tensor of float of shape (N_image, 3, 3) : Rotational matrices randomly generated to orient the configraution during the image generation process
    ##     ctfs_cpu : torch tensor of float of shape (N_image, N_pixel, N_pixel) : random generated CTF added to each of the synthetic image
    ##     images_cpu : torch tensor of float of shape (N_image, N_pixel, N_pixel) : generated synthetic images
    ##

    device = coord.device
    batch_size = coord.shape[0]
    n_struc = coord.shape[0]
    n_atoms = coord.shape[1]
    norm = .5 / (np.pi * sigma ** 2 * n_atoms)
    N_images = n_struc
    n_batch = int(N_images / batch_size)
    if n_batch * batch_size < N_images:
        n_batch += 1

    ref = outil.read_CA_pos(protein_name=protein_name, N_batch=coord.shape[0]).double() * 10
    ref = ref.to(coord.device)
    if coord.shape[1] != ref.shape[1]:
        raise ValueError("pos and ref must have the same number of points")
    R, t = find_alignment_kabsch_batch(coord, ref)
    coord = torch.matmul(R, coord.transpose(1, 2)).transpose(1, 2) + t.unsqueeze(1)

    if rotation:
        quats = gen_quat_torch(N_images, device)
        rot_mats = quaternion_to_matrix(quats).type(torch.float64)
        rot_mats = rot_mats.to(device)
        coord_rot = coord.matmul(rot_mats)
    else:
        rot_mats = torch.eye(3).unsqueeze(0).repeat(N_images, 1, 1).type(torch.float64)
        coord_rot = coord.matmul(rot_mats)
    grid = gen_grid(n_pixel, pixel_size).reshape(-1, 1)
    grid = grid.to(device)

    if add_ctf:
        amp = 0.1  ## Amplitude constrast ratio
        b_factor = 1.0  ## B-factor
        defocus = torch.rand(N_images, dtype=torch.float64, device=device) * (
                    defocus_max - defocus_min) + defocus_min  ## defocus

        elecwavel = 0.019866  ## electron wavelength in Angstrom
        gamma = defocus * (
                    np.pi * 2. * 10000 * elecwavel)  ## gamma coefficient in SI equation 4 that include the defocus

        freq_pix_1d = torch.fft.fftfreq(n_pixel, d=pixel_size, dtype=torch.float64, device=device)
        freq_x, freq_y = torch.meshgrid(freq_pix_1d, freq_pix_1d, indexing='ij')
        freq2_2d = freq_x ** 2 + freq_y ** 2  ## square of modulus of spatial frequency
    if add_ctf:
        ctf_batch = calc_ctf_torch_batch(freq2_2d, amp, gamma, b_factor)
        image_batch = gen_img_torch_batch(coord_rot.to(device), grid, sigma, norm, ctf_batch.to(device))
    else:
        image_batch = gen_img_torch_batch(coord_rot, grid, sigma, norm)
    if not np.isinf(snr):
        image_batch = add_noise_torch_batch(image_batch, snr, device)

    return image_batch.reshape(image_batch.shape[0], -1)