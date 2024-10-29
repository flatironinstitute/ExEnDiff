from einops import repeat, rearrange
import torch
def normalize(v):
    norm = torch.norm(v, dim=-1, keepdim=True)
    norm = torch.where(norm == 0, torch.tensor(1.0, device=v.device), norm)
    return v / norm


def turn(v1, v2, w, d, cos, sin):
    v = v1 - v2
    v = normalize(v)
    w = w - v2
    w = w - torch.sum(v * w, dim=-1, keepdim=True) * v
    w = normalize(w)
    return v2 - d * cos * v + d * sin * w


def completeTetra(v1, v2):
    v1 = normalize(v1)
    v2 = normalize(v2)
    nv = torch.cross(v1, v2, dim=-1)
    nv = 0.8164965809277259 * normalize(nv)
    v = (-v1 - v2) / 2
    w1 = v + nv
    w2 = v - nv
    return w1, w2

def recover_backbone_atoms(ca_positions):
    N_batch, N, _ = ca_positions.size()

    cosg1 = 0.93544403088  # g1 = 20.7 CA CA CO
    sing1 = 0.35347484377
    cosg2 = 0.96944534989  # g2 = 14.2 CA CA N
    sing2 = 0.24530738587
    cos989 = 0.15471038629  # CSC
    sin989 = 0.98795986576
    cos120 = 0.5  # CCO
    sin120 = 0.86602540378  # hexagon
    cos108 = 0.30901699437  # pentagon
    sin108 = 0.95105651629
    cos126 = 0.58778525229  # pentagon external turn
    sin126 = 0.80901699437
    cos1274 = 0.60737583972  # Fe-N-C in HEM
    sin1274 = 0.79441462053
    cos1103 = 0.34693565157  # ? in HEM
    sin1103 = 0.93788893461
    cos1249 = 0.57214587344  # CC-CB-CY in HEM
    sin1249 = 0.82015187587
    cos1166 = 0.44775908783  # CCN
    sin1166 = 0.89415423683
    cos1095 = 0.33380685923
    sin1095 = 0.94264149109
    cos111 = 0.35836794954
    sin111 = 0.93295353482
    cos117 = 0.45399049974  # CT - C - O2
    sin117 = 0.89100652418
    cosd = 0.57714519003  # d = half a tetrahedral angle
    sind = 0.81664155516
    cos45 = 0.70710678118

    dctct = 1.526
    dcc = 1.522
    dcbcc = 1.44400  # HEM
    dcbcy = 1.501  # HEM
    dcycx = 1.34  # HEM
    dco = 1.229
    dcn = 1.335
    dnc = 1.384  # HEM
    dccna = 1.385
    dcccv = 1.375
    dcrna = 1.343
    dcvnb = 1.394

    dctn = 1.449
    dcts = 1.810
    dctoh = 1.410
    dfech = 3.41315894736826
    dfen = 2.01

    n_positions = torch.zeros((N_batch, N, 3))
    c_positions = torch.zeros((N_batch, N, 3))
    o_positions = torch.zeros((N_batch, N, 3))

    ca1 = ca_positions[:, 1:-1] - ca_positions[:, :-2]  # ca1[i] = CA(i+1) - CA(i)
    ca2 = ca_positions[:, 2:] - ca_positions[:, :-2]  # ca2[i] = CA(i+2) - CA(i)
    p_vector = torch.cross(ca1, ca2, dim=-1)  # peptide plane vector
    e1 = normalize(ca1)
    v2 = p_vector - torch.sum(e1 * p_vector, dim=-1, keepdim=True) * e1
    e2 = normalize(v2)
    posC = ca_positions[:, :-2] + dcc * cosg1 * e1 + dcc * sing1 * e2

    posCB_0 = turn(posC[:, :1], ca_positions[:, :1], ca_positions[:, :1] + e2[:, :1], dctct, cos1095, sin1095)
    v0, _ = completeTetra(posC[:, :1] - ca_positions[:, :1], posCB_0 - ca_positions[:, :1])
    posN_0 = ca_positions[:, :1] + dcn * v0
    posN_1_Nm2 = posC - dcn * cos1166 * e2 + dcn * sin1166 * e1
    posO = turn(ca_positions[:, :-2], posC, posN_1_Nm2, dco, cos120, -sin120)
    backbone_atoms = torch.cat(
        [posN_1_Nm2[:, :-1].unsqueeze(2), ca_positions[:, 1:-2].unsqueeze(2), posC[:, 1:].unsqueeze(2),
         posO[:, 1:].unsqueeze(2)], dim=2)
    return backbone_atoms




CONST_Q1Q2 = 0.084
CONST_F = 332
DEFAULT_CUTOFF = -0.5
DEFAULT_MARGIN = 1.0


# DEFAULT_CUTOFF = -8
# DEFAULT_MARGIN = 3.0
def _check_input(coord):
    org_shape = coord.shape
    assert (len(org_shape) == 3) or (
                len(org_shape) == 4), "Shape of input tensor should be [batch, L, atom, xyz] or [L, atom, xyz]"
    coord = repeat(coord, '... -> b ...', b=1) if len(org_shape) == 3 else coord
    return coord, org_shape


def _get_hydrogen_atom_position(coord: torch.Tensor) -> torch.Tensor:
    # A little bit lazy (but should be OK) definition of H position here.
    vec_cn = coord[:, 1:, 0] - coord[:, :-1, 2]
    vec_cn = vec_cn / torch.linalg.norm(vec_cn, dim=-1, keepdim=True)
    vec_can = coord[:, 1:, 0] - coord[:, 1:, 1]
    vec_can = vec_can / torch.linalg.norm(vec_can, dim=-1, keepdim=True)
    vec_nh = vec_cn + vec_can
    vec_nh = vec_nh / torch.linalg.norm(vec_nh, dim=-1, keepdim=True)
    return coord[:, 1:, 0] + 1.01 * vec_nh

def get_hbond_map(
        coord: torch.Tensor,
        cutoff: float = DEFAULT_CUTOFF,
        margin: float = DEFAULT_MARGIN,
        return_e: bool = False
) -> torch.Tensor:
    # check input
    coord, org_shape = _check_input(coord)
    b, l, a, _ = coord.shape
    # add pseudo-H atom if not available
    assert (a == 4) or (a == 5), "Number of atoms should be 4 (N,CA,C,O) or 5 (N,CA,C,O,H)"
    h = coord[:, 1:, 4] if a == 5 else _get_hydrogen_atom_position(coord)
    # distance matrix
    nmap = repeat(coord[:, 1:, 0], '... m c -> ... m n c', n=l - 1)
    hmap = repeat(h, '... m c -> ... m n c', n=l - 1)
    cmap = repeat(coord[:, 0:-1, 2], '... n c -> ... m n c', m=l - 1)
    omap = repeat(coord[:, 0:-1, 3], '... n c -> ... m n c', m=l - 1)
    d_on = torch.linalg.norm(omap - nmap, dim=-1)
    d_ch = torch.linalg.norm(cmap - hmap, dim=-1)
    d_oh = torch.linalg.norm(omap - hmap, dim=-1)
    d_cn = torch.linalg.norm(cmap - nmap, dim=-1)
    # electrostatic interaction energy
    e = torch.nn.functional.pad(CONST_Q1Q2 * (1. / d_on + 1. / d_ch - 1. / d_oh - 1. / d_cn) * CONST_F, [0, 1, 1, 0])
    if return_e: return e
    # mask for local pairs (i,i), (i,i+1), (i,i+2)
    local_mask = ~torch.eye(l, dtype=bool)
    local_mask *= ~torch.diag(torch.ones(l - 1, dtype=bool), diagonal=-1)
    local_mask *= ~torch.diag(torch.ones(l - 2, dtype=bool), diagonal=-2)
    hbond_map = torch.clamp(cutoff - margin - e, min=-margin, max=margin)
    hbond_map = (torch.sin(hbond_map / margin * torch.pi / 2) + 1.) / 2
    hbond_map = hbond_map * repeat(local_mask.to(hbond_map.device), 'l1 l2 -> b l1 l2', b=b)
    # return h-bond map
    hbond_map = hbond_map.squeeze(0) if len(org_shape) == 3 else hbond_map
    return hbond_map


