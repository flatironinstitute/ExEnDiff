import torch
import numpy as np
def gen_grid(n_pixel, pixel_size):
    ##
    ## gen_grid : function : generate square grids of positions of each pixel
    ##
    ## Input:
    ##     n_pixel : int : number of pixels of image i.e. the synthetic (cryo-EM) images are of shape (n_pixel, n_pixel)
    ##     pixel_size : float : width of each pixel in physical space in Angstrom
    ## Output:
    ##     grid : torch tensor of float of shape (N_pixel) : physical location of center of each pixel (in Angstrom)
    ##
    grid_min = -pixel_size * (n_pixel - 1) * 0.5
    grid_max = -grid_min  # pixel_size*(n_pixel-1)*0.5
    grid = torch.linspace(grid_min, grid_max, n_pixel)
    return grid


def gen_quat_torch(num_quaternions, device="cuda"):
    ##
    ## gen_quat_torch : function : sample quaternions from spherically uniform random distribution of directions
    ##
    ## Input:
    ##     num_quaternions: int : number of quaternions generated
    ## Output:
    ##     quat_out : tensor of shape (num_quaternions, 4) : quaternions generated
    ##
    over_produce = 5  ## for ease of parallelizing the calculation, it first produce much more than the needed amount of quanternion, then filter the ones that satisfy the condition
    quat = torch.rand((num_quaternions * over_produce, 4), dtype=torch.float64, device=device) * 2. - 1.
    norm = torch.linalg.vector_norm(quat, ord=2, dim=1)
    quat /= norm.unsqueeze(1)
    good_ones = torch.bitwise_and(torch.gt(norm, 0.2), torch.lt(norm,
                                                                1.0))  ## this condition, norm of quaternion has to be < 1.0 and > 0.2, has to be satisfied
    quat_out = quat[good_ones][:num_quaternions]  ## just chop the ones needed
    return quat_out


def quaternion_to_matrix(quaternions):
    ##
    ## quaternion_to_matrix : function : Convert rotations given as quaternions to rotation matrices
    ##
    ## Input:
    ##     quaternions: tensor of float shape (4) : quaternions leading with the real part
    ## Output:
    ##     rot_mat : tensor of shape (3, 3) : Rotation matrices
    ##
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    rot_mat = o.reshape(quaternions.shape[:-1] + (3, 3))
    return rot_mat


def calc_ctf_torch_batch(freq2_2d, amp, gamma, b_factor):
    ##
    ## calc_ctf_torch_batch : function : generate random Contrast transfer function (CTF)
    ##
    ## Input :
    ##     freq2_2d : torch tensor of float of shape (N_pixel, N_pixel) : square of modulus of spatial frequency in Fourier space
    ##     amp : float : Amplitude constrast ratio
    ##     gamma : torch tensor of float of shape (N_image) : gamma coefficient in SI equation 4 that include the defocus
    ##     b_factor : float : B-factor
    ## Output :
    ##     ctf : torch tensor of float of shape (N_image, N_pixel, N_pixel) : randomly generated CTF
    ##
    # env = torch.exp(- b_factor.view(-1,1,1) * freq2_2d.unsqueeze(0) * 0.5)
    # ctf = amp.view(-1,1,1) * torch.cos(gamma.view(-1,1,1) * freq2_2d * 0.5) - torch.sqrt(1 - amp.view(-1,1,1) **2) * torch.sin(gamma.view(-1,1,1)  * freq2_2d * 0.5) + torch.zeros_like(freq2_2d) * 1j
    env = torch.exp(- b_factor * freq2_2d.unsqueeze(0) * 0.5)
    ctf = amp * torch.cos(gamma.view(-1, 1, 1) * freq2_2d * 0.5) - np.sqrt(1 - amp ** 2) * torch.sin(
        gamma.view(-1, 1, 1) * freq2_2d * 0.5) + torch.zeros_like(freq2_2d) * 1j
    ctf *= env
    return ctf


def gen_img_torch_batch(coord, grid, sigma, norm, ctfs=None):
    ##
    ## gen_img_torch_batch : function : generate images from atomic coordinates
    ##
    ## Input :
    ##     coord : numpy ndarray or torch tensor of float of shape (N_image, N_atom, 3) : 3D Cartesian coordinates of atoms of configuration aligned to generate the synthetic images
    ##     grid : torch tensor of float of shape (N_pixel) : physical location of center of each pixel (in Angstrom)
    ##     sigma : float : Gaussian width of each atom in the imaging model in Angstrom
    ##     norm : float : normalization factor for image intensity
    ##     ctfs : torch tensor of float of shape (N_image, N_pixel, N_pixel) : random generated CTF added to each of the synthetic image
    ## Output :
    ##     image or image_ctf : torch tensor of float of shape (N_image, N_pixel, N_pixel) : synthetic images with or without randomly generated CTF applied
    ##
    gauss_x = -.5 * ((grid[:, :, None] - coord[:, :, 0]) / sigma) ** 2  ##
    gauss_y = -.5 * ((grid[:, :, None] - coord[:, :,
                                         1]) / sigma) ** 2  ## pixels are square, grid is same for x and y directions
    gauss = torch.exp(gauss_x.unsqueeze(1) + gauss_y)
    image = gauss.sum(3) * norm
    image = image.permute(2, 0, 1)
    if ctfs is not None:
        ft_image = torch.fft.fft2(image, dim=(1, 2), norm="ortho")
        image_ctf = torch.real(torch.fft.ifft2(ctfs * ft_image, dim=(1, 2), norm="ortho"))
        return image_ctf
    else:
        return image


def circular_mask(n_pixel, radius=0.4):
    ##
    ## circular_mask : function : define a circular mask centered at center of the image for SNR calculation purpose (see Method for detail)
    ##
    ## Input :
    ##     n_pixel : int : number of pixels of image i.e. the synthetic (cryo-EM) images are of shape (n_pixel, n_pixel)
    ##     radius : float : radius of the circular mask relative to n_pixel, when radius = 0.5, the circular touches the edges of the image
    ## Output :
    ##     mask : torch tensor of bool of shape (N_pixel, N_pixel) : circular mask to be applied onto the image
    ##
    grid = torch.linspace(-.5 * (n_pixel - 1), .5 * (n_pixel - 1), n_pixel)
    grid_x, grid_y = torch.meshgrid(grid, grid, indexing='ij')
    r_2d = grid_x ** 2 + grid_y ** 2
    mask = r_2d < radius ** 2
    return mask


def add_noise_torch_batch(img, snr, device="cuda"):
    ##
    ## add_noise_torch_batch : function : add colorless Gaussian pixel noise to images
    ##
    ## Input :
    ##     n_pixel : int : number of pixels of image i.e. the synthetic (cryo-EM) images are of shape (n_pixel, n_pixel)
    ##     snr : float : Signal-to-noise (SNR) for adding noise to the image, if snr = np.infty, does not add noise to the images
    ## Output :
    ##     image_noise : torch tensor of float of shape (N_image, N_pixel, N_pixel) : synthetic images with added noise
    ##
    n_pixel = img.shape[1]
    radius = n_pixel * 0.4
    mask = circular_mask(n_pixel, radius)
    image_noise = torch.empty_like(img, device=device)
    for i, image in enumerate(img):
        image_masked = image[mask]
        signal_std = image_masked.pow(2).mean().sqrt()
        noise_std = signal_std / np.sqrt(snr)
        noise = torch.distributions.normal.Normal(0, noise_std).sample(image.shape)
        image_noise[i] = image + noise
    return image_noise