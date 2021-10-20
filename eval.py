import os, sys, glob
import cv2
import torch
import numpy as np
from unet import UNet
from tqdm import tqdm
from pathlib import Path
from torch.cuda.amp import autocast
import torchvision.transforms.functional as TF

device = 'cuda:0'

model = UNet().to(device).eval()
model.load_state_dict(torch.load(sys.argv[2], map_location='cpu')['model'])


def normalize(img):
	return (img - 0.5) / 0.5


def denormalize(img):
	return (img * 0.5) + 0.5


def process(file):
	img = cv2.imread(file)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	x = normalize(TF.to_tensor(img))
	x = x.to(device).unsqueeze(0)

	#with autocast(): # enable if amp model
	with torch.inference_mode():
		y = model(x)

	y = denormalize(y[0].cpu())
	y = np.array(TF.to_pil_image(y))

	# optional mask enhancement
	y = cv2.blur(y, (5,5))
	_, y = cv2.threshold(y, 127, 255, cv2.THRESH_BINARY)
	mask = np.zeros_like(y)
	cont, _ = cv2.findContours(y, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cont = cont[np.argmax([len(x) for x in cont])]
	cv2.fillPoly(mask, pts=[cont], color=255)

	cv2.imwrite('masks/%s.png'%Path(file).stem, mask)


def main():

	if len(sys.argv) != 3:
		print('usage: eval.py <data_dir> <checkpoint>')
		exit(1)

	files = sorted(glob.glob(sys.argv[1]+'/*'))

	if not os.path.exists('masks'):
		os.mkdir('masks')

	for file in tqdm(files):
		process(file)


main()
