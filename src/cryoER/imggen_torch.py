import numpy as np
import torch, math
from tqdm import tqdm


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
    grid_min = -pixel_size*(n_pixel-1)*0.5
    grid_max = -grid_min #pixel_size*(n_pixel-1)*0.5
    grid = torch.linspace(grid_min, grid_max, n_pixel)
    return grid

def gen_quat_torch(num_quaternions, device = "cuda"):
    ## 
    ## gen_quat_torch : function : sample quaternions from spherically uniform random distribution of directions
    ## 
    ## Input:
    ##     num_quaternions: int : number of quaternions generated
    ## Output:
    ##     quat_out : tensor of shape (num_quaternions, 4) : quaternions generated
    ## 
    over_produce = 5 ## for ease of parallelizing the calculation, it first produce much more than the needed amount of quanternion, then filter the ones that satisfy the condition
    quat = torch.rand((num_quaternions*over_produce, 4), dtype=torch.float64, device=device) * 2. - 1.
    norm = torch.linalg.vector_norm(quat, ord=2, dim=1)
    quat /= norm.unsqueeze(1)
    good_ones = torch.bitwise_and(torch.gt(norm,0.2), torch.lt(norm,1.0)) ## this condition, norm of quaternion has to be < 1.0 and > 0.2, has to be satisfied
    quat_out  = quat[good_ones][:num_quaternions] ## just chop the ones needed
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
    ctf = amp * torch.cos(gamma.view(-1,1,1) * freq2_2d * 0.5) - np.sqrt(1 - amp**2) * torch.sin(gamma.view(-1,1,1)  * freq2_2d * 0.5) + torch.zeros_like(freq2_2d) * 1j
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
    gauss_x = -.5*((grid[:,:,None]-coord[:,:,0])/sigma)**2  ##
    gauss_y = -.5*((grid[:,:,None]-coord[:,:,1])/sigma)**2  ## pixels are square, grid is same for x and y directions
    gauss = torch.exp(gauss_x.unsqueeze(1) + gauss_y)
    image = gauss.sum(3)*norm
    image = image.permute(2,0,1)
    if ctfs is not None:
        ft_image = torch.fft.fft2(image, dim=(1,2), norm="ortho")
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
    grid = torch.linspace(-.5*(n_pixel-1), .5*(n_pixel-1), n_pixel)
    grid_x, grid_y = torch.meshgrid(grid, grid, indexing='ij')
    r_2d = grid_x**2 + grid_y**2
    mask = r_2d < radius**2
    return mask


def add_noise_torch_batch(img, snr, device = "cuda"):
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
    radius = n_pixel*0.4
    mask = circular_mask(n_pixel, radius)
    image_noise = torch.empty_like(img, device=device)
    for i, image in enumerate(img):
        image_masked = image[mask]
        signal_std = image_masked.pow(2).mean().sqrt()
        noise_std = signal_std / np.sqrt(snr)
        noise = torch.distributions.normal.Normal(0, noise_std).sample(image.shape)
        image_noise[i] = image + noise
    return image_noise


def generate_images(
        coord, 
        n_pixel = 128,  ## use power of 2 for CTF purpose
        pixel_size = 0.3,
        sigma = 1.0, 
        snr = 1.0,
        add_ctf = False,
        batch_size = 8,
        device = "cuda",
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


    if type(coord) == np.ndarray:
        coord = torch.from_numpy(coord).type(torch.float64)
    if device == "cuda":
        coord = coord.cuda()

    n_struc = coord.shape[0]
    n_atoms = coord.shape[1]
    norm = .5/(np.pi*sigma**2*n_atoms)
    N_images = n_struc
    n_batch = int(N_images / batch_size)
    if n_batch * batch_size < N_images:
        n_batch += 1

    quats = gen_quat_torch(N_images, device)
    rot_mats = quaternion_to_matrix(quats).type(torch.float64)
    if device == "cuda":
        rot_mats = rot_mats.cuda()
    coord_rot = coord.matmul(rot_mats)
    grid = gen_grid(n_pixel, pixel_size).reshape(-1,1)
    if device == "cuda":
        grid = grid.cuda()

    ctfs_cpu = torch.empty((N_images, n_pixel, n_pixel), dtype=torch.complex64, device='cpu')
    images_cpu = torch.empty((N_images, n_pixel, n_pixel), dtype=torch.float64, device='cpu')

    if add_ctf:
        amp = 0.1  ## Amplitude constrast ratio 
        b_factor = 1.0  ## B-factor
        defocus = torch.rand(N_images, dtype=torch.float64, device=device) * (0.090 - 0.027) + 0.027  ## defocus
        
        elecwavel = 0.019866  ## electron wavelength in Angstrom
        gamma = defocus * (np.pi * 2. * 10000 * elecwavel) ## gamma coefficient in SI equation 4 that include the defocus

        freq_pix_1d = torch.fft.fftfreq(n_pixel, d=pixel_size, dtype=torch.float64, device=device)
        freq_x, freq_y = torch.meshgrid(freq_pix_1d, freq_pix_1d, indexing='ij')
        freq2_2d = freq_x**2 + freq_y**2  ## square of modulus of spatial frequency
    
    for i in tqdm(range(n_batch)):
        start = i*batch_size
        end = (i+1)*batch_size
        coords_batch = coord_rot[start:end]
        if device == "cuda":
            coords_batch = coords_batch.cuda()
        if add_ctf:
            ctf_batch = calc_ctf_torch_batch(freq2_2d, amp, gamma[start:end], b_factor)
            ctfs_cpu[start:end] = ctf_batch
            image_batch = gen_img_torch_batch(coords_batch, grid, sigma, norm, ctf_batch)
        else:
            image_batch = gen_img_torch_batch(coords_batch, grid, sigma, norm)
        if not np.isinf(snr):
            image_batch = add_noise_torch_batch(image_batch, snr, device)
        images_cpu[start:end] = image_batch.cpu()

    if device == "cuda":
        rot_mats = rot_mats.cpu()
    return rot_mats, ctfs_cpu, images_cpu


def calc_struc_image_diff(
        coord, 
        n_pixel = 128,  ## use power of 2 for CTF purpose
        pixel_size = 0.3,
        sigma = 1.0, 
        images = None,
        ctfs = None,
        batch_size = 8,
        device = "cuda",
    ):
    ## 
    ## calc_struc_image_diff : function : calculate the difference between an image and a structure, given a set of images and a structure (= a set of N_image coordinates of the structure aligned to each of the images)
    ## 
    ## Input :
    ##     coord : numpy ndarray or torch tensor of float of shape (N_image, N_atom, 3) : 3D Cartesian coordinates of atoms of configuration aligned to each of the N_image synthetic images
    ##                 these N_image coordinates represent ONE single configuration that is rotated by N_image different rotation matrices to align the configuration to N_image different images
    ##     n_pixel : int : number of pixels of image i.e. the synthetic (cryo-EM) images are of shape (n_pixel, n_pixel)
    ##     pixel_size : float : width of each pixel in physical space in Angstrom
    ##     sigma : float : Gaussian width of each atom in the imaging model in Angstrom
    ##     images : torch tensor of float of shape (N_image, N_pixel, N_pixel) : Intensity of cryo-EM (synthetic) images to be compared to the structure
    ##     ctfs : torch tensor of float of shape (N_image, N_pixel, N_pixel) :  Contrast transfer function (CTF) for the synthetic images
    ##     batch_size : int : to split the set of images into batches for calculation, where images in the same batch are fed into calculation at the same time, a parameter for computational performance / memory management
    ##     device : str : "cuda" or "cpu", to be fed into pyTorch, see pyTorch manual for more detail
    ## Output :
    ##     diff : torch tensor of float of shape (N_image) : L2-norm distances between the structure (that is converted into images) and the cryo-EM images
    ## 

    if type(coord) == np.ndarray:
        coord = torch.from_numpy(coord).type(torch.float64)
        if device == "cuda":
            coord = coord.cuda()

    n_atoms = coord.shape[1]
    norm = .5/(np.pi*sigma**2*n_atoms)
    N_images = coord.shape[0]
    n_batch = int(N_images / batch_size)
    if n_batch * batch_size < N_images:
        n_batch += 1

    grid = gen_grid(n_pixel, pixel_size).reshape(-1,1)
    if device == "cuda":
        grid = grid.cuda()

    diff = torch.empty(N_images, dtype=torch.float64, device='cpu')
    
    for i in range(n_batch):
        start = i*batch_size
        end = (i+1)*batch_size
        if ctfs is not None:
            ctf_batch = ctfs[start:end]
            if device == "cuda":
                ctf_batch = ctf_batch.cuda()
            image_batch = gen_img_torch_batch(coord[start:end], grid, sigma, norm, ctf_batch)
        else:
            image_batch = gen_img_torch_batch(coord[start:end], grid, sigma, norm)
        image_batch = image_batch - image_batch.mean(dim=(1,2)).view(-1,1,1)
        if device == "cuda":
            diff[start:end] = torch.sum((image_batch - images[start:end].cuda())**2, dim=(1,2)).cpu()
        else:
            diff[start:end] = torch.sum((image_batch - images[start:end])**2, dim=(1,2))

    return diff



