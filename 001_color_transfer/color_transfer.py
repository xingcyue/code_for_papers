import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def get_mean_and_std(x):
	x_mean, x_std = cv.meanStdDev(x)
	x_mean = np.around(x_mean,2).reshape(1,3)
	x_std = np.around(x_std,2).reshape(1,3)
	return x_mean, x_std


def histogram(img):
	for i in range(3):
		plt.subplot(1,3,i+1), plt.title("channel: " + str(i))
		plt.hist(img[:,:,i].reshape(-1), bins=256, range=(0,256), density=True)
	plt.show()


def color_transfer():
	for i in range(1,7):
		s = cv.imread("./source/s" + str(i) + ".bmp")
		t = cv.imread("./target/t" + str(i) + ".bmp")
		
		# plt.subplot(131), plt.title("source"), plt.imshow(s[:,:,::-1])
		# plt.subplot(132), plt.title("target"), plt.imshow(t[:,:,::-1])
		s = cv.cvtColor(s, cv.COLOR_BGR2LAB)
		t = cv.cvtColor(t, cv.COLOR_BGR2LAB)
		s_mean, s_std = get_mean_and_std(s)
		t_mean, t_std = get_mean_and_std(t)
		r = ((s - s_mean) * (t_std / (s_std)) + t_mean)
		r = np.clip(r, 0, 255).astype(np.uint8)
		# histogram(s)
		# histogram(t)
		# histogram(r)
		r = cv.cvtColor(r, cv.COLOR_LAB2BGR)
		
		cv.imwrite("./result/r" + str(i) + "_bgr.bmp", r)
		
		# plt.subplot(133), plt.title("result"), plt.imshow(r[:,:,::-1])
		# plt.show()
		

color_transfer()
