import random
from model import *
import torch
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

rgb_mean = torch.tensor([0.5, 0.5, 0.5])
rgb_std = torch.tensor([0.5, 0.5, 0.5])
transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize(rgb_mean, rgb_std)
])

vgg19 = models.vgg19(pretrained=True).features.to(device)
for param in vgg19.parameters():
	param.requires_grad = False
style_layers = [1, 6, 11, 20, 28]
# style_layers = [1, 3, 6, 8, 11, 13, 15, 17, 20, 22, 24, 26]
style_weights = [1e3 / n ** 2 for n in [64, 128, 256, 512, 512]]


def imshow(tensor, title=None):
	image = tensor.cpu().clone()
	image = image.squeeze(0)
	print(image.shape)
	image = torch.clamp(image.permute(1, 2, 0) * rgb_std + rgb_mean, 0, 1)
	image = transforms.ToPILImage()(image.permute(2, 0, 1))
	plt.imshow(image)
	if title is not None:
		plt.title(title)
	plt.pause(0.01)
	return image


def extract_features(X):
	styles = []
	for i in range(len(vgg19)):
		X = vgg19[i](X)
		if i in style_layers:
			styles.append(X)
	return styles


def clac_gram_loss(list_A, list_B):
	loss = 0
	for feature_A, feature_B, style_weight in zip(list_A, list_B, style_weights):
		b, c, h, w = feature_A.shape
		A = feature_A.reshape(b, c, h * w)
		B = feature_B.reshape(b, c, h * w)
		gram_A = torch.bmm(A, A.transpose(1, 2)).div(h * w)
		gram_B = torch.bmm(B, B.transpose(1, 2)).div(h * w)

		loss += F.mse_loss(gram_A, gram_B.detach()) * style_weight
	return loss


def clac_swd_loss(list_A, list_B):
	loss = 0
	for feature_A, feature_B, style_weight in zip(list_A, list_B, style_weights):
		b, c, w, h = feature_A.shape
		A = feature_A.reshape(b, c, w * h)
		B = feature_B.reshape(b, c, w * h)
		Vr = torch.randn((c, c), device=device)
		norm = Vr.norm(dim=1, keepdim=True)
		Vr = Vr / norm
		sliced_A, _ = (Vr @ A).sort(dim=2)
		sliced_B, _ = (Vr @ B).sort(dim=2)

		loss += F.mse_loss(sliced_A, sliced_B.detach()) * style_weight
	return loss


def main():
	img = Image.open('./data/5.jpg').convert('RGB')
	test_img = transform(img).unsqueeze(0).to(device)
	w, h = img.size
	fineSize = 256
	cropSize = 128
	gen_net = Generator().to(device)
	disc_net = Discriminator().to(device)
	gen_net.apply(weights_init)
	disc_net.apply(weights_init)
	optimizer_G = torch.optim.Adam(gen_net.parameters(), lr=0.0002, betas=(0.5, 0.999))
	optimizer_D = torch.optim.Adam(disc_net.parameters(), lr=0.0002, betas=(0.5, 0.999))

	real_label = torch.ones((1, 1, 14, 14), requires_grad=False).to(device)
	fake_label = torch.zeros((1, 1, 14, 14), requires_grad=False).to(device)

	for epoch in tqdm(range(1, 15001)):
		rw = random.randint(0, w - fineSize)
		rh = random.randint(0, h - fineSize)
		target_img = img.crop((rw, rh, rw + fineSize, rh + fineSize))

		rw = random.randint(0, cropSize)
		rh = random.randint(0, cropSize)
		input_img = target_img.crop((rw, rh, rw + cropSize, rh + cropSize))

		input_img = transform(input_img).unsqueeze(0).to(device)
		target_img = transform(target_img).unsqueeze(0).to(device)
		target_features = extract_features(target_img)

		# 生成器
		optimizer_G.zero_grad()
		gen_img = gen_net(input_img)
		pred_fake = disc_net(gen_img)
		loss_G_GAN = F.mse_loss(pred_fake, real_label)

		gen_features = extract_features(gen_img)
		loss_style = clac_gram_loss(gen_features, target_features)
		# loss_style = clac_swd_loss(gen_features, target_features)

		loss_G_L1 = F.l1_loss(target_img, gen_img)
		loss_G = loss_G_GAN + loss_G_L1 + 10.0 * loss_style
		loss_G.backward()
		optimizer_G.step()

		# 判别器
		optimizer_D.zero_grad()
		pred_fake = disc_net(gen_img.detach())
		loss_D_fake = F.mse_loss(pred_fake, fake_label)
		pred_real = disc_net(target_img)
		loss_D_real = F.mse_loss(pred_real, real_label)
		loss_D = (loss_D_fake + loss_D_real) * 0.5
		loss_D.backward()
		optimizer_D.step()

		with torch.no_grad():
			if epoch % 1000 == 0:
				print("")
				print(loss_G_GAN)
				# print(loss_G_L1)
				print(loss_style)
				test_img_2x = gen_net(test_img)
				image1 = imshow(gen_img, title=f'epoch: {epoch}')
				image2 = imshow(test_img_2x, title=f'epoch: {epoch}_2x')
				image3 = imshow(gen_net(test_img_2x), title=f'epoch: {epoch}_4x')

				path1 = f'./result/epoch_{epoch}.jpg'
				path2 = f'./2x/epoch_{epoch}.jpg'
				path3 = f'./4x/epoch_{epoch}.jpg'
				image1.save(path1)
				image2.save(path2)
				image3.save(path3)


if __name__ == '__main__':
	main()
