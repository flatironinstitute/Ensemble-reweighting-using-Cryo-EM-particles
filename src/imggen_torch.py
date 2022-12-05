import numpy as np
import torch, math
from tqdm import tqdm

def calc_ctf_torch(freq2_2d, amp, phase, b_factor):
    env = torch.exp(- b_factor * freq2_2d * 0.5)
    ctf = amp * torch.cos(phase * freq2_2d * 0.5) - torch.sqrt(1 - amp**2) * torch.sin(phase * freq2_2d * 0.5) + torch.zeros_like(freq2_2d) * 1j
    return ctf * env 

# def add_noise_torch(img, snr):
#     std_image = torch.std(img)
#     mask = torch.abs(img) > 0.5 * std_image
#     signal_mean = torch.mean(img[mask])
#     signal_std = torch.std(img[mask])
#     noise_std = signal_std / math.sqrt(snr)
#     noise = torch.normal(signal_mean, noise_std, size=img.shape, device="cuda")
#     img_noise = img + noise
#     img_noise -= torch.mean(img_noise)
#     img_noise /= torch.std(img_noise)
#     return img_noise


def gen_grid(n_pixels, pixel_size):
    grid_min = -pixel_size*(n_pixels-1)*0.5
    grid_max = -grid_min #pixel_size*(n_pixels-1)*0.5
    grid = torch.linspace(grid_min, grid_max, n_pixels)
    return grid


def gen_img_torch(coord, grid, sigma, norm, ctf):
    gauss_x = -.5*((grid-coord[0,:])/sigma)**2
    gauss_y = -.5*((grid-coord[1,:])/sigma)**2
    gauss = torch.exp(gauss_x.unsqueeze(1) + gauss_y)
    image = gauss.sum(2)*norm
    ft_image = torch.fft.fft2(image, norm="ortho")
    image_ctf = torch.real(torch.fft.ifft2(ctf * ft_image, norm="ortho")) 
    return image_ctf 


def gen_quat_torch(num_quaternions, device = "cuda"):
    #Sonya's code
    over_produce = 5
    quat = torch.rand((num_quaternions*over_produce, 4), dtype=torch.float64, device=device) * 2. - 1.
    norm = torch.linalg.vector_norm(quat, ord=2, dim=1)
    quat /= norm.unsqueeze(1)
    good_ones = torch.bitwise_and(torch.gt(norm,0.2), torch.lt(norm,1.0))
    return quat[good_ones][:num_quaternions]


def quaternion_to_matrix(quaternions):
    """ Convert rotations given as quaternions to rotation matrices. Args: quaternions: quaternions with real part first, as tensor of shape (..., 4). Returns: Rotation matrices as tensor of shape (..., 3, 3). """
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
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def calc_ctf_torch_batch(freq2_2d, amp, phase, b_factor):
    env = torch.exp(- b_factor.view(-1,1,1) * freq2_2d.unsqueeze(0) * 0.5)
    ctf = amp.view(-1,1,1) * torch.cos(phase.view(-1,1,1) * freq2_2d * 0.5) - torch.sqrt(1 - amp.view(-1,1,1) **2) * torch.sin(phase.view(-1,1,1)  * freq2_2d * 0.5) + torch.zeros_like(freq2_2d) * 1j
    ctf *= env
    return ctf

def gen_img_torch_batch(coord, grid, sigma, norm, ctf=None):
    gauss_x = -.5*((grid[:,:,None]-coord[:,:,0])/sigma)**2
    gauss_y = -.5*((grid[:,:,None]-coord[:,:,1])/sigma)**2
    gauss = torch.exp(gauss_x.unsqueeze(1) + gauss_y)
    image = gauss.sum(3)*norm
    image = image.permute(2,0,1)
    if ctf is not None:
        ft_image = torch.fft.fft2(image, dim=(1,2), norm="ortho")
        image_ctf = torch.real(torch.fft.ifft2(ctf * ft_image, dim=(1, 2), norm="ortho"))
        return image_ctf
    else:
        return image

# def add_noise_torch_batch(img, snr):
#     std_image = torch.std(img, dim=(1,2))
#     mask = torch.abs(img) > 0.5 * std_image.view(-1,1,1)
#     mask_count = torch.sum(mask, dim=(1,2))
#     signal_mean = torch.sum(img*mask, dim=(1,2))/mask_count
#     signal_std = torch.std(img*mask, dim=(1,2))
#     noise_std = signal_std / math.sqrt(snr)
#     noise = torch.distributions.normal.Normal(signal_mean, noise_std).sample(img[0].shape).permute(2,0,1)
#     img_noise = img + noise
#     img_noise -= torch.mean(img_noise, dim=(1,2)).view(-1,1,1)
#     img_noise /= torch.std(img_noise, dim=(1,2)).view(-1,1,1)
#     return img_noise

# def add_noise_torch_batch(img, snr):
#     num_images = img.shape[0]
#     centered_images = img - img.mean(dim=(1,2)).view(-1,1,1)
#     std_images = centered_images.std(dim=(1,2)) # std of image intensity
#     noise_stds = torch.empty(num_images)
#     for i in range(num_images):
#         mask = torch.abs(centered_images[i]).ge(0.5*std_images[i])  # mask image with intensity > 0.5*std
#         masked_image = torch.masked_select(centered_images[i], mask) # mask image with intensity > 0.5*std
#         signal_std = torch.std(masked_image) # std pixel intensity (sigma_signal)
#         noise_stds[i] = signal_std / math.sqrt(snr)
#     noise = torch.distributions.normal.Normal(0, noise_stds).sample(img[0].shape).permute(2,0,1).cuda()
#     img_noise = img + noise
#     img_noise = img_noise - img_noise.mean(dim=(1,2)).view(-1,1,1)
#     return img_noise

def circular_mask(n_pixels, radius):
    grid = torch.linspace(-.5*(n_pixels-1), .5*(n_pixels-1), n_pixels)
    grid_x, grid_y = torch.meshgrid(grid, grid, indexing='ij')
    r_2d = grid_x**2 + grid_y**2
    mask = r_2d < radius**2
    return mask

def add_noise_torch_batch(img, snr, device = "cuda"):
    n_pixels = img.shape[1]
    radius = n_pixels*0.4
    mask = circular_mask(n_pixels, radius)
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
        num_images_per_struc = 1, 
        n_pixels = 128,  ## use power of 2 for CTF purpose
        pixel_size = 0.3,
        sigma = 1.0, 
        snr = 1.0,
        ctf = False,
        batch_size = 8,
        device = "cuda",
    ):

    if type(coord) == np.ndarray:
        coord = torch.from_numpy(coord).type(torch.float64).cuda()
    if coord.device == "cpu":
        coord = coord.cuda()

    n_struc = coord.shape[0]
    n_atoms = coord.shape[1]
    norm = .5/(np.pi*sigma**2*n_atoms)
    num_images = num_images_per_struc * n_struc
    n_batch = int(num_images / batch_size)
    if n_batch * batch_size < num_images:
        n_batch += 1

    quats = gen_quat_torch(num_images, device)
    if device == "cuda":
        rot_mats = quaternion_to_matrix(quats).type(torch.float64).cuda()
    else:
        rot_mats = quaternion_to_matrix(quats).type(torch.float64)
    coord_rot = coord.matmul(rot_mats)
    grid = gen_grid(n_pixels, pixel_size).reshape(-1,1)

    ctfs_cpu = torch.empty((num_images, n_pixels, n_pixels), dtype=torch.complex64, device='cpu')
    images_cpu = torch.empty((num_images, n_pixels, n_pixels), dtype=torch.float64, device='cpu')

    if ctf:
        b_factor = torch.rand(num_images, dtype=torch.float64, device=device)
        defocus = torch.rand(num_images, dtype=torch.float64, device=device) * (3.0 - 0.9) + 0.9
        amp = torch.rand(num_images, dtype=torch.float64, device=device)
        elecwavel = 0.019866
        phase  = defocus * (np.pi * 2. * 300 * elecwavel)

        freq_pix_1d = torch.fft.fftfreq(n_pixels, d=pixel_size, dtype=torch.float64, device=device)
        freq_x, freq_y = torch.meshgrid(freq_pix_1d, freq_pix_1d, indexing='ij')
        freq2_2d = freq_x**2 + freq_y**2
    
    for i in tqdm(range(n_batch)):
        start = i*batch_size
        end = (i+1)*batch_size
        if num_images_per_struc > 1:
            if device == "cuda":
                coords_batch = coord_rot[int(start/num_images_per_struc):int(end/num_images_per_struc)].cuda()
            else:
                coords_batch = coord_rot[int(start/num_images_per_struc):int(end/num_images_per_struc)]
        else:
            if device == "cuda":
                coords_batch = coord_rot[start:end].cuda()
            else:
                coords_batch = coord_rot[start:end]
        if ctf:
            ctf_batch = calc_ctf_torch_batch(freq2_2d, amp[start:end], phase[start:end], b_factor[start:end])
            ctfs_cpu[start:end] = ctf_batch
            image_batch = gen_img_torch_batch(coords_batch, grid, sigma, norm, ctf_batch)
        else:
            image_batch = gen_img_torch_batch(coords_batch, grid, sigma, norm)
        if not np.isinf(snr):
            image_batch = add_noise_torch_batch(image_batch, snr, device)
        images_cpu[start:end] = image_batch.cpu()

    return rot_mats.cpu(), ctfs_cpu, images_cpu


def calc_struc_image_diff(
        coord, 
        n_pixels = 128,  ## use power of 2 for CTF purpose
        pixel_size = 0.3,
        sigma = 1.0, 
        images = None,
        ctfs = None,
        batch_size = 8,
        device = "cuda",
    ):

    if type(coord) == np.ndarray:
        if device == "cuda":
            coord = torch.from_numpy(coord).type(torch.float64).cuda()
        else:
            coord = torch.from_numpy(coord).type(torch.float64)

    n_atoms = coord.shape[1]
    norm = .5/(np.pi*sigma**2*n_atoms)
    num_images = coord.shape[0]
    n_batch = int(num_images / batch_size)
    if n_batch * batch_size < num_images:
        n_batch += 1

    grid = gen_grid(n_pixels, pixel_size).reshape(-1,1)
    if device == "cuda":
        grid = grid.cuda()

    diff = torch.empty(num_images, dtype=torch.float64, device='cpu')
    
    for i in range(n_batch):
        start = i*batch_size
        end = (i+1)*batch_size
        if ctfs is not None:
            if device == "cuda":
                ctf_batch = ctfs[start:end].cuda()
            else:
                ctf_batch = ctfs[start:end]
            image_batch = gen_img_torch_batch(coord[start:end], grid, sigma, norm, ctf_batch)
        else:
            image_batch = gen_img_torch_batch(coord[start:end], grid, sigma, norm)
        image_batch = image_batch - image_batch.mean(dim=(1,2)).view(-1,1,1)
        if device == "cuda":
            diff[start:end] = torch.sum((image_batch - images[start:end].cuda())**2, dim=(1,2)).cpu()
        else:
            diff[start:end] = torch.sum((image_batch - images[start:end])**2, dim=(1,2))

    return diff



