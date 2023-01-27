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

class Model:
	def __init__(self, model_name, model):
		self.model_name = model_name
		self.model = model
	
	def get_model(self, name: str):
		return self.model if self.model_name == name else None
		

optimized_model = Model(None, None)

start_datetime_string = datetime.strftime(datetime.now(), '%d.%m.%Y %H-%M-%S')

if not os.path.isdir("reports"):
	os.makedirs("reports")
if not os.path.isdir("predicted_images"):
	os.makedirs("predicted_images")
os.makedirs(f"predicted_images/{start_datetime_string}")

print_depth = 0
logger_last_end_str = "\n"
report_file_name = f"reports/image_recognition_report_{start_datetime_string}.txt"
open(report_file_name, "w").close()
avg_time_by_model = {}


def get_str_depth():
	global print_depth
	return "\t"*print_depth

def logger(*args, end="\n"):
	global logger_last_end_str, report_file_name
	default_name = '\n'
	print(get_str_depth(), *args, end=end)
	timestamp = f"{datetime.now()} | "
	logger_last_end_str = end
	with open(report_file_name, "a") as file:
		file.write(f"{timestamp if logger_last_end_str == default_name else ''}{get_str_depth()} {' '.join([str(arg) for arg in args])}{end}")

def preprocess_image(image):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (config.IMAGE_SIZE, config.IMAGE_SIZE))
	image = image.astype("float32") / 255.0
	image -= config.MEAN
	image /= config.STD
	image = np.transpose(image, (2, 0, 1))
	image = np.expand_dims(image, 0)
	return image

def classify_image(model, current_image: str, is_tr_model: bool = False, current_model_name: str = "")-> Tuple[Optional[Tensor], Optional[Tensor], Optional[Any], Optional[float]]:
	global optimized_model
	logger(f"loading image:", current_image, "...", end="")
	image = cv2.imread(f"images/{current_image}")
	if image is None:
		logger(f"\n		Image {current_image} not found!")
		return None, None, None, None
	orig = image.copy()
	image = preprocess_image(image)
	image = torch.from_numpy(image)
	image = image.to(config.DEVICE)
	logger("Done")
	if is_tr_model:
		model_trt = optimized_model.get_model(current_model_name)
		if model_trt is None:
			model_trt = torch2trt(model, [image], use_onnx=False)
			logger(f"Optimyze model {current_model_name}...")
			optimized_model = Model(current_model_name, model_trt)
		else:
			logger(f"Get optimyzed model {current_model_name} from storege")
		if model_trt is None:
			logger(f"Fail to optimize model {current_model_name}. Skip")
			return None, None, None, None
		logger(f"Optimization complete")
		logger(f"Classifying image with trt '{current_model_name}'...", end="")
		start_time = time.time()
		logits = model_trt(image)
		end_time = time.time()
		logger("Done")
	else:
		logger(f"Classifying image with '{current_model_name}'...", end="")
		start_time = time.time()
		logits = model(image)
		end_time = time.time()
		logger("Done")
	probabilities = nn.Softmax(dim=-1)(logits)
	sortedProba = torch.argsort(probabilities, dim=-1, descending=True)
	return sortedProba, probabilities, orig, end_time-start_time

images_names = os.listdir("images")

MODELS = {
	#"vgg16": models.vgg16(pretrained=True),
	# "vgg19": models.vgg19(pretrained=True),
	#"inception": models.inception_v3(pretrained=True),
	"densenet": models.densenet121(pretrained=True),
	"resnet": models.resnet50(pretrained=True),
	"mobilenet": models.mobilenet_v3_small(pretrained=True)
}

models_keys = list(MODELS.keys())

images_names_sample = random.choices(images_names, k=10)
logger("images sample:", images_names_sample)
logger(f"Using device {config.DEVICE}")
logger(f"loading ImageNet labels...")
imagenetLabels = dict(enumerate(open(config.IN_LABELS)))
print_depth +=1
for current_model_name in models_keys:
	logger(f"loading model {current_model_name}...")
	os.makedirs(f"predicted_images/{start_datetime_string}/{current_model_name}")
	model = MODELS[current_model_name].to(config.DEVICE)
	model.eval()
	print_depth +=1
	counter = 0
	total_time = 0
	total_optimize_time = 0
	for current_image in images_names_sample:
		counter+=1
		for is_tr_model in [False, True]:  # if True, then use torch2trt
			if is_tr_model:
				pass
			start_time = time.time()
			try:
				sortedProba, probabilities, orig, time_pass = classify_image(model, current_image, is_tr_model, current_model_name)
			except Exception:
				continue
			if sortedProba is None:
				continue
			end_time = time.time()
			if is_tr_model:
				total_optimize_time += time_pass
			else:
				total_time += time_pass
			logger(f"Time passed {time_pass}. Image {current_image} result")
			print_depth +=1
			for (i, idx) in enumerate(sortedProba[0, :5]):
				logger(f"{i}. {imagenetLabels[int(idx.item())].strip()}: {probabilities[0, idx.item()] * 100}")
			print_depth -=1
			(label, prob) = (imagenetLabels[probabilities.argmax().item()],
                            probabilities.max().item())
			cv2.putText(orig, f"Label: {label.strip()}, {prob * 100}% Time :{time_pass} sec",
						(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			cv2.imwrite(f"predicted_images/{start_datetime_string}/{current_model_name}/prediction_{current_image}.jpg", orig)
		# cv2.imshow("Classification", orig)
		# cv2.waitKey(0)
	print_depth -= 1
	logger(f"total_time: {total_time}. avg: {total_time/counter}. avg optimize: {total_optimize_time/counter}")
	avg_time_by_model[current_model_name] = {"avg": total_time/counter, "avg_optimize": total_optimize_time/counter}
print_depth -=1

avg_time_by_model = dict(sorted(avg_time_by_model.items(), key=lambda item: item[1]["avg"] if item[1]["avg"]<item[1]["avg_optimize"] else item[1]["avg_optimize"]))
logger("-----------------\nRESUME")
print_depth+=1
logger("model: avg")
for model, avg in avg_time_by_model.items():
	logger(f"{model}: {avg}")
