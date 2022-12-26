from pyimagesearch import config
from torchvision import models
import numpy as np
import argparse
import torch
import cv2
from torch import nn
import time
import os
import random

def preprocess_image(image):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (config.IMAGE_SIZE, config.IMAGE_SIZE))
	image = image.astype("float32") / 255.0
	image -= config.MEAN
	image /= config.STD
	image = np.transpose(image, (2, 0, 1))
	image = np.expand_dims(image, 0)
	return image


images_names = os.listdir("images")

MODELS = {
	"vgg16": models.vgg16(pretrained=True),
	"vgg19": models.vgg19(pretrained=True),
	"inception": models.inception_v3(pretrained=True),
	"densenet": models.densenet121(pretrained=True),
	"resnet": models.resnet50(pretrained=True)
}

models_keys = list(MODELS.keys())

print_depth = 0

print(f"{'\t'*print_depth}[INFO] loading ImageNet labels...")
imagenetLabels = dict(enumerate(open(config.IN_LABELS)))
print(f"{'\t'*print_depth}[INFO] loading {models_keys[0]}...")
model = MODELS[models_keys[0]].to(config.DEVICE)
model.eval()
print_depth +=1
for current_image in random.choices(images_names, k=10):
	print(f"{'\t'*print_depth}[INFO] loading image:", current_image)
	image = cv2.imread(current_image)
	orig = image.copy()
	image = preprocess_image(image)
	image = torch.from_numpy(image)
	image = image.to(config.DEVICE)

	print(f"{'\t'*print_depth}[INFO] classifying image with '{models_keys[0]}'...")
	logits = model(image)
	probabilities = nn.Softmax(dim=-1)(logits)
	sortedProba = torch.argsort(probabilities, dim=-1, descending=True)
	print(f"{'\t'*print_depth}Image {current_image} result")
	print_depth +=1
	for (i, idx) in enumerate(sortedProba[0, :5]):
		print(f"{'\t'*print_depth}{i}. {imagenetLabels[int(idx.item())].strip()}: {probabilities[0, idx.item()] * 100}")
	print_depth -=1
	(label, prob) = (imagenetLabels[probabilities.argmax().item()],
		probabilities.max().item())
	# cv2.putText(orig, "Label: {}, {:.2f}%".format(label.strip(), prob * 100),
	# 	(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
	# cv2.imshow("Classification", orig)
	# cv2.waitKey(0)
print_depth -=1