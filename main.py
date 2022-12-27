from datetime import datetime
from torchvision import models
from torch2trt import torch2trt
import numpy as np
import torch
from torch import Tensor
import cv2
from torch import nn
import time
import os
import random
from typing import Any, Optional, Tuple
import config


start_datetime_string = datetime.strftime(datetime.now(), '%d.%m.%Y %H-%M-%S')

if not os.path.isdir("reports"):
	os.makedirs("reports")
if not os.path.isdir("prediction_images"):
	os.makedirs("prediction_images")
os.makedirs(f"prediction_images/{start_datetime_string}")

print_depth = 0
report_file_name = f"reports/image_recognition_report_{start_datetime_string}.txt"
open(report_file_name, "w").close()
avg_time_by_model = {}


def get_str_depth():
	global print_depth
	return "\t"*print_depth

def logger(*args):
	global result_file, report_file_name
	print(get_str_depth(), *args)
	with open(report_file_name, "a") as file:
		file.write(f"{datetime.now()} | {get_str_depth()} {' '.join([str(arg) for arg in args])}\n")

def preprocess_image(image):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (config.IMAGE_SIZE, config.IMAGE_SIZE))
	image = image.astype("float32") / 255.0
	image -= config.MEAN
	image /= config.STD
	image = np.transpose(image, (2, 0, 1))
	image = np.expand_dims(image, 0)
	return image

def classify_image(model, current_image: str, is_tr_model: bool = False)-> Tuple[Optional[Tensor], Optional[Any], Optional[Any]]:
	logger(f"loading image:", current_image)
	image = cv2.imread(f"images/{current_image}")
	if image is None:
		logger(f"Image {current_image} not found!")
		return None, None, None
	orig = image.copy()
	image = preprocess_image(image)
	image = torch.from_numpy(image)
	image = image.to(config.DEVICE)
	if is_tr_model:
		logger(f"Optimyze model {current_model}...")
		model_trt = torch2trt(model, image, use_onnx=True)
		logits = model_trt(image)
		logger(f"classifying image with trt '{current_model}'...")
	else:
		logger(f"classifying image with '{current_model}'...")
		logits = model(image)
	probabilities = nn.Softmax(dim=-1)(logits)
	sortedProba = torch.argsort(probabilities, dim=-1, descending=True)
	return sortedProba, probabilities, orig

images_names = os.listdir("images")

MODELS = {
	"vgg16": models.vgg16(pretrained=True),
	"vgg19": models.vgg19(pretrained=True),
	"inception": models.inception_v3(pretrained=True),
	"densenet": models.densenet121(pretrained=True),
	"resnet": models.resnet50(pretrained=True)
}

models_keys = list(MODELS.keys())

images_names_sample = random.choices(images_names, k=10)
logger("images sample:", images_names_sample)
logger(f"Using device {config.DEVICE}")
logger(f"loading ImageNet labels...")
imagenetLabels = dict(enumerate(open(config.IN_LABELS)))
print_depth +=1
for current_model in [models_keys[1]]:
	logger(f"loading model {current_model}...")
	os.makedirs(f"prediction_images/{start_datetime_string}/{current_model}")
	model = MODELS[current_model].to(config.DEVICE)
	model.eval()
	print_depth +=1
	counter = 0
	total_time = 0
	for current_image in images_names_sample:
		counter+=1
		for is_tr_model in [False, True]:  # if True, then use torch2trt
			if is_tr_model:
				pass
			start_time = time.time()
			sortedProba, probabilities, orig = classify_image(model, current_image, is_tr_model)
			if sortedProba is None:
				continue
			end_time = time.time()
			total_time += end_time-start_time
			logger(f"Time passed {end_time-start_time}. Image {current_image} result")
			print_depth +=1
			for (i, idx) in enumerate(sortedProba[0, :5]):
				logger(f"{i}. {imagenetLabels[int(idx.item())].strip()}: {probabilities[0, idx.item()] * 100}")
			print_depth -=1
			(label, prob) = (imagenetLabels[probabilities.argmax().item()],
					probabilities.max().item())
			cv2.putText(orig, f"Label: {label.strip()}, {prob * 100}% Time :{end_time-start_time} sec",
						(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			cv2.imwrite(f"prediction_images/{start_datetime_string}/{current_model}/prediction_{current_image}.jpg", orig)
		# cv2.imshow("Classification", orig)
		# cv2.waitKey(0)
	print_depth -= 1
	logger(f"total_time: {total_time}. avg: {total_time/counter}")
	avg_time_by_model[current_model] = total_time/counter
print_depth -=1

logger("-----------------\nRESUME")
print_depth+=1
logger("model: avg")
for model, avg in avg_time_by_model.items():
	logger(f"{model}: {avg}")