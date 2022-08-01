from patch_swd import PatchSWDLoss
from utils import *

img_path = "images/33.jpg"
init_from = "zeros"
patch_size = 11
stride = 1
lr = 0.01
num_step = 300
aspect_ratio = (1, 1)
pyramid_factor = 0.85
coarse_dim = 21
num_proj = 64
additive_noise_sigma = 1.5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

reference_img = Image.open(img_path).convert("RGB")
reference_img = transforms.ToTensor()(reference_img).unsqueeze(0).to(device) * 2.0 - 1.0
synthesized_img = get_first_initial_guess(reference_img, init_from, additive_noise_sigma).to(device)
b, c, h, w = synthesized_img.shape
pyramid_scales = get_pyramid_scales(h, coarse_dim, pyramid_factor)
print(pyramid_scales)

for scale in pyramid_scales:
    print(f"Current scale: {scale}")
    lvl_reference = transforms.Resize(scale, antialias=True)(reference_img)
    lvl_output_shape = get_output_shape(h, w, scale, aspect_ratio)
    synthesized_img = transforms.Resize(lvl_output_shape, antialias=True)(synthesized_img)
    criteria = PatchSWDLoss(patch_size, stride, num_proj).to(device)
    synthesized_img = match_patch_distributions(synthesized_img, lvl_reference, criteria, num_step, lr)

dump_images(synthesized_img, "output")







