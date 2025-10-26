from segment_anything import SamAutomaticMaskGenerator, sam_model_registry,SamPredictor
import cv2
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import glob
from skimage import morphology
from tensorboard_logger import configure, log_value
from skimage.metrics import structural_similarity as ssim


def get_candidate_lines(input_masks):
	# sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
	img = np.full((input_masks[0].shape[0], input_masks[0].shape[1]), False)
	for index,m in enumerate(input_masks):
	    index = index+1
	    img[m] = True
	img = remove_spots(img)
	masks = cv2.Laplacian(img,cv2.CV_64F)
	masks[np.where(masks<0)] = 0
	masks[np.where(masks>0)] = 255 
	return masks

def process_gt(gt_mask):
	return cv2.cvtColor(gt_mask, cv2.COLOR_BGR2GRAY)//255

def get_label(json_dir,label_type='point'):
	points = []
	labels = []
	data = open(json_dir)
	d = json.load(data)
	if label_type == 'point':
		for tmp in d['shapes']:
			if tmp['shape_type'] == "point":
				points.append(tmp['points'][0])
				labels.append(1)
		return np.array(points),np.array(labels)
	else:
		for tmp in d['shapes']:
			if tmp['shape_type'] == "rectangle":
			# if tmp['shape_type'] == "polygon":
				points.append(tmp['points'])
				labels.append(1)
		return points

def is_in_range(x,y,ranges):
	for rangei in ranges:
		if x <= rangei[0][0] or x >= rangei[1][0] or y <= rangei[0][1] or y >= rangei[1][1]:
			continue
		else:
			return True
	return False

def remove_spots(masks):
	masks = morphology.remove_small_objects(masks, 10000)
	masks = morphology.remove_small_holes(masks, 10000)
	return np.float64(255*masks)
	

image_list = glob.glob('/home/yuanfeng/Fire_Project/Data/FireData/20191008FireFlight2NIR_RawImage4Stich/selected/images/*.png')


for image_dir in image_list:
	# json_dir = image_dir.replace('.jpg','.json').replace('/images/','/labels/')

	json_dir = image_dir.replace('.png','.json')
	points,labels = get_label(json_dir)
	sam = sam_model_registry["vit_b"](checkpoint="./checkpoints/sam_vit_b_01ec64.pth").to('cuda')
	predictor = SamPredictor(sam)
	
	image = cv2.imread(image_dir)
	
	#step 1, get segmentations from sams
	# sam_results = mask_generator.generate(image)
	predictor.set_image(image)
	sam_results, _, _ = predictor.predict(
		point_coords=points,
        point_labels=labels,
        multimask_output=False,
		)

	#step 2, get border lines.
	masks = get_candidate_lines(sam_results)
	sam_border = masks.copy()
	ranges = get_label(json_dir,'rectangle')

	h,w = masks.shape
	for x in range(h):
		for y in range(w):
			if not is_in_range(y,x,ranges):
				masks[x][y] = 0
	save_dir = image_dir.replace('/images/','/labels/')
	result_image_dir = image_dir.replace('/images/','/result_image_label/')

	# # show the label on original image (overlay)
	result_image = image
	for r in range(h):
		for c in range(w):
			if masks[r,c] == 255:
				result_image[r,c] = np.array([0,0,0])
				# print(r, c, result_image[r,c])

	masks = cv2.cvtColor( np.float32(masks),cv2.COLOR_GRAY2RGB)
	cv2.imwrite(save_dir,masks)
	cv2.imwrite(result_image_dir,result_image)

	print('DONE！！！')



