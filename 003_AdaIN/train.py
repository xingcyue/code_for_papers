import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from dataset import PreprocessDataset, denorm
from model import *
import torchvision.models as models
import torch.nn.functional as F

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
print(device)
vgg19 = models.vgg19(pretrained=True).features.to(device)
for param in vgg19.parameters():
	param.requires_grad = False
style_layers = [1, 6, 11, 20]
content_layers = [20]


def extract_features(x):
	styles = []
	contents = []
	for i in range(len(vgg19)):
		x = vgg19[i](x)
		if i in style_layers:
			styles.append(x)
		if i in content_layers:
			contents.append(x)
	return styles, contents


def calc_mean_std(features):
	"""

	:param features: shape of features -> [batch_size, c, h, w]
	:return: features_mean, feature_s: shape of mean/std ->[batch_size, c, 1, 1]
	"""

	batch_size, c = features.size()[:2]
	features_mean = features.reshape(batch_size, c, -1).mean(dim=2).reshape(batch_size, c, 1, 1)
	features_std = features.reshape(batch_size, c, -1).std(dim=2).reshape(batch_size, c, 1, 1) + 1e-6
	return features_mean, features_std


def adain(content_t, style_t):
	"""
	Adaptive Instance Normalization

	:param content_t: shape -> [batch_size, c, h, w]
	:param style_t: shape -> [batch_size, c, h, w]
	:return: t shape -> [batch_size, c, h, w]
	"""
	content_mean, content_std = calc_mean_std(content_t)
	style_mean, style_std = calc_mean_std(style_t)
	t = style_std * (content_t - content_mean) / content_std + style_mean
	return t


def calc_content_loss(content_t, t):
	return F.mse_loss(content_t, t)


def clac_style_loss(list_A, list_B):
	loss = 0
	for feature_A, feature_B in zip(list_A, list_B):
		A_mean, A_std = calc_mean_std(feature_A)
		B_mean, B_std = calc_mean_std(feature_B)
		loss += F.mse_loss(A_mean, B_mean.detach()) + F.mse_loss(A_std, B_std.detach())
	return loss


def main():
	image_dir = './result/image'
	loss_dir = './result/loss'
	model_state_dir = './result/model_state'

	content_dir = './dataset/content'
	style_dir = './dataset/style'

	batch_size = 8
	epoch_num = 10

	# prepare data_set and dataLoader
	data_set = PreprocessDataset(content_dir, style_dir)
	print(f'Length of train image pairs: {len(data_set)}')
	data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True, num_workers=16)

	# set model and optimizer
	model = Decoder().to(device)
	print(model)
	optimizer = Adam(params=model.parameters(), lr=1e-4, betas=(0.5, 0.9))

	# start training
	loss_list = []
	for e in range(1, epoch_num + 1):
		print(f'Start {e} epoch')
		for i, (content, style) in tqdm(enumerate(data_loader, 1)):
			content = content.to(device)
			style = style.to(device)

			_, content_t = extract_features(content)
			_, style_t = extract_features(style)
			t = adain(*content_t, *style_t).detach()
			output = model(t)

			output_style, output_content = extract_features(output)
			_, content_content = extract_features(content)
			style_style, _ = extract_features(style)

			loss_content = calc_content_loss(*output_content, *content_content)
			loss_style = clac_style_loss(output_style, style_style)

			loss = loss_content + 10.0 * loss_style
			loss_list.append(loss.item())

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			with torch.no_grad():
				if i % 1000 == 0:
					content = denorm(content, device)
					style = denorm(style, device)
					out = denorm(output, device)
					res = torch.cat([content, style, out], dim=0)
					res = res.to('cpu')
					save_image(res, f'{image_dir}/{e}_epoch_{i}_iteration.png', nrow=batch_size)
		torch.save(model.state_dict(), f'{model_state_dir}/{e}_epoch.pth')
	plt.plot(range(len(loss_list)), loss_list)
	plt.xlabel('iteration')
	plt.ylabel('loss')
	plt.title('train loss')
	plt.savefig(f'{loss_dir}/train_loss.png')


if __name__ == '__main__':
	main()
