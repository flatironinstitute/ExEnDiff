import numpy as np
import scipy.ndimage
import torch


def sample_secondary_structure(P_H, P_E, P_C, alpha_H=0.9, alpha_E=0.9, min_length=3, smoothing_sigma=2):
    N = len(P_H)
    S = ['C'] * N  # Initialize all residues as coil
    assigned = [False] * N

    # Step 1: Smooth the per-residue probabilities
    # Apply Gaussian filter to P_H and P_E
    P_H_smooth = scipy.ndimage.gaussian_filter1d(P_H, sigma=smoothing_sigma)
    P_E_smooth = scipy.ndimage.gaussian_filter1d(P_E, sigma=smoothing_sigma)

    # After smoothing, ensure probabilities are between 0 and 1
    P_H_smooth = np.clip(P_H_smooth, 0, 1)
    P_E_smooth = np.clip(P_E_smooth, 0, 1)

    # Recompute coil probabilities
    P_C_smooth = 1 - P_H_smooth - P_E_smooth

    # Correct any negative probabilities due to smoothing
    P_C_smooth = np.clip(P_C_smooth, 0, 1)

    # Normalize probabilities to sum to 1 at each position
    total_probs = P_H_smooth + P_E_smooth + P_C_smooth
    P_H_smooth /= total_probs
    P_E_smooth /= total_probs
    P_C_smooth /= total_probs

    # Step 2: Identify peaks in the smoothed probabilities
    helix_peaks = find_peaks(P_H_smooth)
    beta_peaks = find_peaks(P_E_smooth)

    # Step 3: Proceed with sampling using smoothed probabilities
    # Process helix peaks
    for i in helix_peaks:
        if assigned[i]:
            continue
        # Sample at peak
        S_i = np.random.choice(['H', 'C'], p=[P_H_smooth[i], 1 - P_H_smooth[i]])
        if S_i == 'H':
            S[i] = 'H'
            assigned[i] = True
            # Forward propagation
            j = i + 1
            while j < N:
                if assigned[j]:
                    break
                P_cont = alpha_H * P_H_smooth[j]
                P_term = (1 - alpha_H) * (1 - P_H_smooth[j])
                total = P_cont + P_term
                P_cont /= total
                P_term /= total
                S_j = np.random.choice(['H', 'C'], p=[P_cont, P_term])
                if S_j == 'H':
                    S[j] = 'H'
                    assigned[j] = True
                    j += 1
                else:
                    break
            # Backward propagation
            j = i - 1
            while j >= 0:
                if assigned[j]:
                    break
                P_cont = alpha_H * P_H_smooth[j]
                P_term = (1 - alpha_H) * (1 - P_H_smooth[j])
                total = P_cont + P_term
                P_cont /= total
                P_term /= total
                S_j = np.random.choice(['H', 'C'], p=[P_cont, P_term])
                if S_j == 'H':
                    S[j] = 'H'
                    assigned[j] = True
                    j -= 1
                else:
                    break

    # Process beta peaks
    for i in beta_peaks:
        if assigned[i]:
            continue
        # Sample at peak
        S_i = np.random.choice(['E', 'C'], p=[P_E_smooth[i], 1 - P_E_smooth[i]])
        if S_i == 'E':
            S[i] = 'E'
            assigned[i] = True
            # Forward propagation
            j = i + 1
            while j < N:
                if assigned[j]:
                    break
                P_cont = alpha_E * P_E_smooth[j]
                P_term = (1 - alpha_E) * (1 - P_E_smooth[j])
                total = P_cont + P_term
                P_cont /= total
                P_term /= total
                S_j = np.random.choice(['E', 'C'], p=[P_cont, P_term])
                if S_j == 'E':
                    S[j] = 'E'
                    assigned[j] = True
                    j += 1
                else:
                    break
            # Backward propagation
            j = i - 1
            while j >= 0:
                if assigned[j]:
                    break
                P_cont = alpha_E * P_E_smooth[j]
                P_term = (1 - alpha_E) * (1 - P_E_smooth[j])
                total = P_cont + P_term
                P_cont /= total
                P_term /= total
                S_j = np.random.choice(['E', 'C'], p=[P_cont, P_term])
                if S_j == 'E':
                    S[j] = 'E'
                    assigned[j] = True
                    j -= 1
                else:
                    break

    # Step 4: Filter short helices and beta strands
    S = filter_short_structures(S, min_length=min_length)

    return S, P_H_smooth, P_E_smooth, P_C_smooth


def find_peaks(P):
    peaks = []
    N = len(P)
    for i in range(1, N - 1):
        if P[i] > P[i - 1] and P[i] > P[i + 1]:
            peaks.append(i)
    return peaks


def filter_short_structures(S, min_length=3):
    N = len(S)
    i = 0
    while i < N:
        current = S[i]
        if current in ['H', 'E']:
            start = i
            while i < N and S[i] == current:
                i += 1
            length = i - start
            if length < min_length:
                # Convert to coil
                for j in range(start, i):
                    S[j] = 'C'
        else:
            i += 1
    return S


def sample_N(P_H, P_E, P_C, num_samples=1000, alpha_H=1, alpha_E=1):
    secondary_structures = []
    secondary_structures_filter = []
    for _ in range(num_samples):
        S, P_H_smooth, P_E_smooth, P_C_smooth = sample_secondary_structure(P_H, P_E, P_C, alpha_H=alpha_H,
                                                                           alpha_E=alpha_E)
        secondary_structures.append(S)
        S1 = filter_short_structures(S, min_length=3)
        secondary_structures_filter.append(S1)
    return secondary_structures, secondary_structures_filter


def compute_freq(secondary_structures):
    N = len(secondary_structures[0])
    num_sample = len(secondary_structures)
    freq_H = np.zeros(N)
    freq_E = np.zeros(N)
    freq_C = np.zeros(N)
    for S in secondary_structures:
        for i, s in enumerate(S):
            if s == 'H':
                freq_H[i] += 1
            elif s == 'E':
                freq_E[i] += 1
            elif s == 'C':
                freq_C[i] += 1

    # Convert counts to frequencies
    F_H = freq_H / num_sample
    F_E = freq_E / num_sample
    F_C = freq_C / num_sample
    return F_H, F_E, F_C


def calc_nrmsd(rmsd, constant=0.1):
    return (1 - (rmsd / constant) ** 8) / (1 - (rmsd / constant) ** 12)


def calc_rmsd(pos, ref, constant=0.1):
    ref = ref.unsqueeze(0).repeat(pos.shape[0], 1, 1)
    rmsd = calculate_rmsd_batch(pos, ref)
    nrmsd = calc_nrmsd(rmsd, constant)
    return nrmsd


def helix_func1(B_tensor_pair, pair_mask, pair_indices_tensor, N_dim, ref=torch.randn(6, 3)):
    N_batch, num_cons = B_tensor_pair.shape[0], B_tensor_pair.shape[1]
    output = torch.zeros(N_batch, num_cons, N_dim).to(B_tensor_pair.device)
    B_tensor_pair = B_tensor_pair.view(N_batch * num_cons, 6, 3)
    output1 = (calc_rmsd(B_tensor_pair, ref).view(N_batch, num_cons, 1) * pair_mask.unsqueeze(-1))
    output1 = output1.repeat(1, 1, 6)
    # print(pair_indices_tensor.dtype,output1.dtype,output.dtype)
    output.scatter_(dim=2, index=pair_indices_tensor, src=output1)

    sum_over_dim = output.sum(dim=1, keepdim=True)  # Shape: N_batch * 1 * M

    # Step 2: Count non-zero elements along the N_dim dimension
    nonzero_count = (output != 0).sum(dim=1, keepdim=True).float()  # Shape: N_batch * 1 * M

    # Step 3: Avoid division by zero by replacing zero counts with 1 (to prevent NaNs)
    nonzero_count[nonzero_count == 0] = 1

    # Step 4: Divide the sum by the non-zero count
    result = sum_over_dim / nonzero_count  # Shape: N_batch * 1 * M

    # Squeeze to remove the extra dimension if needed (optional)
    result = result.squeeze(1)  # Shape: N_batch * M
    return result
    # return torch.sum(output,dim = (1))


def beta_func1(B_tensor_pair, pair_mask, pair_indices_tensor, N_dim, ref=torch.randn(6, 3)):
    N_batch, num_cons = B_tensor_pair.shape[0], B_tensor_pair.shape[1]
    output = torch.zeros(N_batch, num_cons, num_cons, N_dim).to(B_tensor_pair.device)
    B_tensor_pair = B_tensor_pair.view(N_batch * num_cons * num_cons, 6, 3)
    output1 = (calc_rmsd(B_tensor_pair, ref, constant=0.1).view(N_batch, num_cons, num_cons, 1) * pair_mask.unsqueeze(
        -1))
    output1 = output1.repeat(1, 1, 1, 6)
    output.scatter_(dim=3, index=pair_indices_tensor, src=output1)
    max_over_dims = output.max(dim=1)[0].max(dim=1)[0]
    return max_over_dims



def get_alpha_indice(A):
    # Step 1.1: Identify consecutive triplets of 1s in A
    N_batch, N_dim = A.shape[0], A.shape[1]
    is_one = (A == 0)
    consecutive_ones = is_one[:, 0:-5] & is_one[:, 1:-4] & is_one[:, 2:-3] & is_one[:, 3:-2] & is_one[:, 4:-1] & is_one[
                                                                                                                 :, 5:]

    # Get the indices of consecutive triplets
    triplet_indices = torch.arange(N_dim - 5).unsqueeze(0).expand(N_batch, -1).to(
        A.device)  # Shape: N_batch * (N_dim - 2)
    triplet_starts = torch.where(consecutive_ones, triplet_indices, -1).to(
        A.device)  # Replace non-triplet positions with -1

    # Sort the triplet starts so that padded values (-1) go to the end
    sorted_triplet_starts, _ = triplet_starts.sort(dim=1, descending=True)

    # Get the number of valid triplets in each batch
    num_triplets_per_batch = (sorted_triplet_starts >= 0).sum(dim=1)

    # Pad triplet indices to the maximum number of triplets across batches
    max_consecutive_6 = num_triplets_per_batch.max().item()  # Max number of triplets across all batches
    padded_triplet_starts = sorted_triplet_starts[:, :max_consecutive_6]  # Shape: N_batch * max_consecutive_3

    # Step 1.2: Create the final triplet indices (N_batch * N_consecutive * 3)
    triplet_indices_tensor = torch.stack([padded_triplet_starts,
                                          padded_triplet_starts + 1,
                                          padded_triplet_starts + 2,
                                          padded_triplet_starts + 3,
                                          padded_triplet_starts + 4,
                                          padded_triplet_starts + 5], dim=-1)  # Shape: N_batch * max_consecutive_3 * 3

    # Replace invalid triplet indices with [-3, -2, -1]
    triplet_indices_tensor[triplet_indices_tensor[:, :, 0] == -1] = torch.tensor(
        [N_dim - 6, N_dim - 5, N_dim - 4, N_dim - 3, N_dim - 2, N_dim - 1]).to(A.device)

    first_triplet_expanded = triplet_indices_tensor.unsqueeze(2).expand(-1, -1, triplet_indices_tensor.size(1),
                                                                        -1)  # Shape: N_batch * N_consecutive * N_consecutive * 3

    # Expand the second triplet to match pairs
    second_triplet_expanded = triplet_indices_tensor.unsqueeze(1).expand(-1, triplet_indices_tensor.size(1), -1,
                                                                         -1)  # Shape: N_batch * N_consecutive * N_consecutive * 3

    # The second triplet must start at least 3 indices after the first triplet ends
    first_triplet_ends = triplet_indices_tensor[:, :, 2]  # End index of the first triplet
    second_triplet_starts = triplet_indices_tensor[:, :, 0]  # Start index of the second triplet

    # Create the mask tensor
    mask_tensor = triplet_indices_tensor[:, :, 0] >= 0  # Mask for valid triplets

    return mask_tensor, triplet_indices_tensor


def get_alpha_position(B, indices):
    # B is of shape N_batch * N_dim * N_features (e.g., N_batch * N_dim * 3)
    # indices is of shape N_batch * N_pair * m, where m are indices in the N_dim of B

    # Get the shape parameters
    N_batch, N_pair, m = indices.shape
    N_dim, N_features = B.shape[1], B.shape[2]  # N_dim and N_features from B

    # Create the batch indices for advanced indexing
    batch_indices = torch.arange(N_batch).view(N_batch, 1, 1).expand(N_batch, N_pair, m).to(
        B.device)  # Shape: N_batch * N_pair * m

    # Use the batch indices and the provided indices to gather the correct values from B
    gathered_values = B[batch_indices, indices]  # Shape: N_batch * N_pair * m * N_features

    return gathered_values


def get_beta_indice(A):
    # Step 1.1: Identify consecutive triplets of 1s in A
    N_batch, N_dim = A.shape[0], A.shape[1]
    is_one = (A == 1)
    consecutive_ones = is_one[:, :-2] & is_one[:, 1:-1] & is_one[:, 2:]

    # Get the indices of consecutive triplets
    triplet_indices = torch.arange(N_dim - 2).unsqueeze(0).expand(N_batch, -1).to(
        A.device)  # Shape: N_batch * (N_dim - 2)
    triplet_starts = torch.where(consecutive_ones, triplet_indices, -1).to(
        A.device)  # Replace non-triplet positions with -1

    # Sort the triplet starts so that padded values (-1) go to the end
    sorted_triplet_starts, _ = triplet_starts.sort(dim=1, descending=True)

    # Get the number of valid triplets in each batch
    num_triplets_per_batch = (sorted_triplet_starts >= 0).sum(dim=1)

    # Pad triplet indices to the maximum number of triplets across batches
    max_consecutive_3 = num_triplets_per_batch.max().item()  # Max number of triplets across all batches
    padded_triplet_starts = sorted_triplet_starts[:, :max_consecutive_3]  # Shape: N_batch * max_consecutive_3

    # Step 1.2: Create the final triplet indices (N_batch * N_consecutive * 3)
    triplet_indices_tensor = torch.stack([padded_triplet_starts,
                                          padded_triplet_starts + 1,
                                          padded_triplet_starts + 2], dim=-1)  # Shape: N_batch * max_consecutive_3 * 3

    # Replace invalid triplet indices with [-3, -2, -1]
    triplet_indices_tensor[triplet_indices_tensor[:, :, 0] == -1] = torch.tensor([N_dim - 3, N_dim - 2, N_dim - 1]).to(
        A.device)

    first_triplet_expanded = triplet_indices_tensor.unsqueeze(2).expand(-1, -1, triplet_indices_tensor.size(1),
                                                                        -1)  # Shape: N_batch * N_consecutive * N_consecutive * 3

    # Expand the second triplet to match pairs
    second_triplet_expanded = triplet_indices_tensor.unsqueeze(1).expand(-1, triplet_indices_tensor.size(1), -1,
                                                                         -1)  # Shape: N_batch * N_consecutive * N_consecutive * 3

    # The second triplet must start at least 3 indices after the first triplet ends
    first_triplet_ends = triplet_indices_tensor[:, :, 2]  # End index of the first triplet
    second_triplet_starts = triplet_indices_tensor[:, :, 0]  # Start index of the second triplet
    pair_indices_tensor = torch.cat([first_triplet_expanded, second_triplet_expanded], dim=-1)
    # Create the mask tensor
    mask_tensor = triplet_indices_tensor[:, :, 0] >= 0  # Mask for valid triplets
    pair_mask = (first_triplet_ends.unsqueeze(2) + 3 <= second_triplet_starts.unsqueeze(1)) & \
                mask_tensor.unsqueeze(2) & mask_tensor.unsqueeze(1)  # Shape: N_batch * N_consecutive * N_consecutive

    return pair_mask, triplet_indices_tensor, pair_indices_tensor


def get_beta_position(B, triplet_indices_tensor):
    N_batch = B.shape[0]
    batch_indices = torch.arange(N_batch).unsqueeze(1).unsqueeze(1).expand(-1, triplet_indices_tensor.size(1), 3).to(
        B.device)
    # Advanced indexing to get the values from B
    B_tensor = B[batch_indices, triplet_indices_tensor]  # Shape: N_batch * N_consecutive * 3 * 3
    # print(B.shape,batch_indices.shape,triplet_indices_tensor.shape,B_tensor.shape)
    # Step 2: Create B_tensor_pair (N_batch * N_consecutive * N_consecutive * 6 * 3)
    # Pairing first and second triplets by concatenating
    B_tensor_pair = torch.cat([B_tensor.unsqueeze(2).expand(-1, -1, B_tensor.size(1), -1, -1),
                               B_tensor.unsqueeze(1).expand(-1, B_tensor.size(1), -1, -1, -1)],
                              dim=-2)  # Shape: N_batch * N_consecutive * N_consecutive * 6 * 3
    # print(batch_indices.shape, B_tensor.shape,B_tensor_pair.shape)
    return B_tensor_pair


def create_A(N_batch, N_dim):
    A = torch.full((N_batch, N_dim), 2)  # Initialize the entire tensor with 2

    for i in range(N_batch):
        # Generate two non-overlapping positions for six consecutive 0s
        valid_positions_for_zeros = list(range(0, N_dim - 6))
        zero_start_idx_1 = valid_positions_for_zeros[torch.randint(0, len(valid_positions_for_zeros), (1,)).item()]
        # Remove invalid positions for the second zero sequence to avoid overlap
        valid_positions_for_zeros_2 = list(range(0, zero_start_idx_1)) + list(range(zero_start_idx_1 + 6, N_dim - 6))
        zero_start_idx_2 = valid_positions_for_zeros_2[torch.randint(0, len(valid_positions_for_zeros_2), (1,)).item()]

        # Place the consecutive six 0s
        A[i, zero_start_idx_1:zero_start_idx_1 + 6] = 0
        A[i, zero_start_idx_2:zero_start_idx_2 + 6] = 0

        # Generate three non-overlapping positions for three consecutive 1s
        valid_positions_for_ones = list(range(0, N_dim - 3))
        # Remove invalid positions for the three 1 sequences to avoid overlap with the 0s
        valid_positions_for_ones = [pos for pos in valid_positions_for_ones if not (
                zero_start_idx_1 <= pos < zero_start_idx_1 + 6 or zero_start_idx_2 <= pos < zero_start_idx_2 + 6
        )]

        one_start_idx_1 = valid_positions_for_ones[torch.randint(0, len(valid_positions_for_ones), (1,)).item()]
        valid_positions_for_ones = [pos for pos in valid_positions_for_ones if not (
                one_start_idx_1 <= pos < one_start_idx_1 + 3
        )]
        one_start_idx_2 = valid_positions_for_ones[torch.randint(0, len(valid_positions_for_ones), (1,)).item()]
        valid_positions_for_ones = [pos for pos in valid_positions_for_ones if not (
                one_start_idx_2 <= pos < one_start_idx_2 + 3
        )]
        one_start_idx_3 = valid_positions_for_ones[torch.randint(0, len(valid_positions_for_ones), (1,)).item()]

        # Place the consecutive three 1s
        A[i, one_start_idx_1:one_start_idx_1 + 3] = 1
        A[i, one_start_idx_2:one_start_idx_2 + 3] = 1
        A[i, one_start_idx_3:one_start_idx_3 + 3] = 1

    return A


def find_alignment_kabsch_batch(P, Q):
    """Find alignment using Kabsch algorithm between two sets of points P and Q for batches.

    Args:
    P (torch.Tensor): A tensor of shape (B, N, 3) representing the first set of points in batches.
    Q (torch.Tensor): A tensor of shape (B, N, 3) representing the second set of points in batches.

    Returns:
    Tuple[Tensor, Tensor]: A tuple containing two tensors, where the first tensor is the rotation matrix R
    and the second tensor is the translation vector t. The rotation matrix R is a tensor of shape (B, 3, 3)
    representing the optimal rotation for each batch, and the translation vector t is a tensor of shape (B, 3)
    representing the optimal translation for each batch.

    """
    B, N, _ = P.shape
    # Shift points w.r.t centroid
    centroid_P = P.mean(dim=1, keepdim=True)
    centroid_Q = Q.mean(dim=1, keepdim=True)
    P_c, Q_c = P - centroid_P, Q - centroid_Q

    # Find rotation matrix by Kabsch algorithm
    H = torch.matmul(P_c.transpose(1, 2), Q_c)
    U, S, Vt = torch.linalg.svd(H)
    V = Vt.transpose(1, 2)

    # ensure right-handedness
    d = torch.sign(torch.linalg.det(torch.matmul(V, U.transpose(1, 2))))

    diag_values = torch.cat([
        torch.ones(B, 1, dtype=P.dtype, device=P.device),
        torch.ones(B, 1, dtype=P.dtype, device=P.device),
        d.unsqueeze(1)
    ], dim=1)

    M = torch.eye(3, dtype=P.dtype, device=P.device).unsqueeze(0).repeat(B, 1, 1)
    M[:, range(3), range(3)] = diag_values

    R = torch.matmul(V, torch.matmul(M, U.transpose(1, 2)))

    # Find translation vectors
    t = centroid_Q - torch.matmul(R, centroid_P.transpose(1, 2)).transpose(1, 2)

    return R, t.squeeze(1)


def calculate_rmsd_batch(pos, ref):
    """
    Calculate the root mean square deviation (RMSD) between two sets of points pos and ref for batches.

    Args:
    pos (torch.Tensor): A tensor of shape (B, N, 3) representing the positions of the first set of points in batches.
    ref (torch.Tensor): A tensor of shape (B, N, 3) representing the positions of the second set of points in batches.

    Returns:
    torch.Tensor: RMSD between the two sets of points for each batch.

    """
    if pos.shape[1] != ref.shape[1]:
        raise ValueError("pos and ref must have the same number of points")
    R, t = find_alignment_kabsch_batch(ref, pos)
    ref0 = torch.matmul(R, ref.transpose(1, 2)).transpose(1, 2) + t.unsqueeze(1)
    rmsd = torch.linalg.norm(ref0 - pos, dim=2).mean(dim=1).view(-1, 1)
    return rmsd


def read_sec_struc(file_path, seq_len):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Use a more flexible search for the start and end indices
    start_index = next(i for i, line in enumerate(lines) if
                       line.strip() == '#Populations per residue (residues marked with a * are less reliable):') + 2
    end_index = next(i for i, line in enumerate(lines) if line.strip() == '#DONE!')

    # Initialize a list to store the data
    result = np.ones((seq_len, 4)) * -1
    list_availble = []
    # Loop through the relevant lines
    max_indice = 0
    sequence = 'A' * seq_len
    for line in lines[start_index:end_index]:
        parts = line.strip().split()  # Strip whitespace and split the line
        # print(parts)
        if "#" in parts[0]:
            index = int(parts[0][1:])
        else:
            index = int(parts[0])
        index = index - 1
        list_availble.append(index)
        sequence = sequence[:index] + parts[1] + sequence[index + 1:]
        if len(parts) >= 6:  # Ensure there are at least 6 columns (Helix, Beta, Coil, PPII, SS)
            if "#" in parts[0]:
                index = int(parts[0][1:])
            else:
                index = int(parts[0])
            # list_availble.append(index)
            helix = float(parts[2]) if parts[2] else '-1'
            beta = float(parts[3]) if parts[3] else '-1'
            coil = float(parts[4]) if parts[4] else '-1'
            ppii = float(parts[5]) if parts[5] else '-1'
            result[index, 0] = helix
            result[index, 1] = beta
            result[index, 2] = coil
            result[index, 3] = ppii
            # sequence = sequence[:index] + parts[1] + sequence[index + 1:]
            if index > max_indice:
                max_indice = index
    top_10 = np.sort(result[:, 1])[-10:][::-1]

    print("Top 10 maximum values are:", top_10)
    return result, max_indice, sequence, list_availble


def interpolate(arr):
    # Sample array with some values and -1 for missing values
    # Get indices where the value is not -1
    arr = np.concatenate(([0], arr, [0]))
    non_neg_indices = np.where(arr != -1)[0]
    # Get the values at those indices
    non_neg_values = arr[non_neg_indices]

    # Perform linear interpolation
    arr_interpolated = np.interp(np.arange(len(arr)), non_neg_indices, non_neg_values)

    return arr_interpolated

def find_consecutive_six(tensor):
    N_batch, N_dim = tensor.shape
    device = tensor.device  # Get the device of the input tensor
    result_indices = []
    result_mask = []

    for batch_idx in range(N_batch):
        current_batch = tensor[batch_idx]
        indices = []
        mask = []
        count = 0
        start_idx = None

        # Track positions of consecutive 1s
        for i in range(N_dim):
            if current_batch[i] == 1:
                if start_idx is None:
                    start_idx = i
                count += 1
            else:
                if count >= 6:
                    # Apply logic for storing the first indices of blocks of 6
                    for j in range(0, count, 6):
                        if j + 6 <= count:  # Full blocks of 6
                            indices.append(list(range(start_idx + j, start_idx + j + 6)))
                            mask.append(1)  # Not padded
                    if count % 6 != 0 and count > 6:
                        indices.append(list(range(start_idx + count - 6, start_idx + count)))
                        mask.append(1)  # Not padded
                count = 0
                start_idx = None

        # Edge case: if it ends with a sequence of 1s
        if count >= 6:
            for j in range(0, count, 6):
                if j + 6 <= count:  # Full blocks of 6
                    indices.append(list(range(start_idx + j, start_idx + j + 6)))
                    mask.append(1)  # Not padded
            if count % 6 != 0 and count > 6:
                indices.append(list(range(start_idx + count - 6, start_idx + count)))
                mask.append(1)  # Not padded

        result_indices.append(indices)
        result_mask.append(mask)

    # Find max length of the indices array for padding
    max_len = max(len(batch_indices) for batch_indices in result_indices)

    # Pad all batches to the same length with [0,1,2,3,4,5] for indices and 0 for mask
    padded_indices = []
    padded_mask = []
    for batch_indices, batch_mask in zip(result_indices, result_mask):
        while len(batch_indices) < max_len:
            batch_indices.append([0, 1, 2, 3, 4, 5])  # Padding with [0, 1, 2, 3, 4, 5]
            batch_mask.append(0)  # Padded
        padded_indices.append(batch_indices)
        padded_mask.append(batch_mask)

    # Convert to tensors and move to the same device as the input tensor
    padded_indices = torch.tensor(padded_indices, device=device)
    padded_mask = torch.tensor(padded_mask, device=device)

    return padded_indices, padded_mask

def read_CA_pos(protein_name="rs_peptide",N_batch = 64):
    pdb_file = "data/folded_structure/{}.pdb".format(protein_name)
    traj = md.load(pdb_file)
    alpha_carbons = traj.topology.select('name CA')
    alpha_carbon_positions = traj.xyz[:, alpha_carbons, :]
    alpha_carbon_positions_reshaped = alpha_carbon_positions.reshape(-1, 3)
    repeated_array = np.tile(alpha_carbon_positions_reshaped, (N_batch, 1, 1))
    tensor = torch.tensor(repeated_array)
    return tensor
