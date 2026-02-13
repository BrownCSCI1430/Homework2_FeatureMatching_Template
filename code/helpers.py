# Helper functions for the feature matching pipeline.
# You don't need to change anything here, but reading the DINOv3
# section at the bottom will help you understand how it works.

import os
import scipy.io as scio
import skimage
import numpy as np
import visualize
import matplotlib.pyplot as plt
import math

# Gives you the TA solution for the feature points you
# should find
def cheat_feature_points(eval_file, scale_factor):

	file_contents = scio.loadmat(eval_file)

	x1 = file_contents['x1']
	y1 = file_contents['y1']
	x2 = file_contents['x2']
	y2 = file_contents['y2']

	x1 = x1 * scale_factor
	y1 = y1 * scale_factor
	x2 = x2 * scale_factor
	y2 = y2 * scale_factor

	x1 = x1.reshape(-1).astype(int)
	y1 = y1.reshape(-1).astype(int)
	x2 = x2.reshape(-1).astype(int)
	y2 = y2.reshape(-1).astype(int)

	return x1, y1, x2, y2

def estimate_fundamental_matrix(ground_truth_correspondence_file):
    F_path = ground_truth_correspondence_file[:-3] + 'npy'
    return np.load(F_path)

def evaluate_correspondence(img_A, img_B, ground_truth_correspondence_file,
	scale_factor, x1_est, y1_est, x2_est, y2_est, matches, filename="notre_dame_matches.jpg"):

	# 'unscale' feature points to compare with ground truth points
	x1_est_scaled = x1_est / scale_factor
	y1_est_scaled = y1_est / scale_factor
	x2_est_scaled = x2_est / scale_factor
	y2_est_scaled = y2_est / scale_factor

	# we want to see how good our matches are, extract the coordinates of each matched
	# point

	x1_matches = np.zeros(matches.shape[0])
	y1_matches = np.zeros(matches.shape[0])
	x2_matches = np.zeros(matches.shape[0])
	y2_matches = np.zeros(matches.shape[0])

	for i in range(matches.shape[0]):

		x1_matches[i] = x1_est_scaled[int(matches[i, 0])]
		y1_matches[i] = y1_est_scaled[int(matches[i, 0])]
		x2_matches[i] = x2_est_scaled[int(matches[i, 1])]
		y2_matches[i] = y2_est_scaled[int(matches[i, 1])]

	good_matches = np.zeros((matches.shape[0]), dtype=bool)

	# Loads `ground truth' positions x1, y1, x2, y2
	file_contents = scio.loadmat(ground_truth_correspondence_file)

	# x1, y1, x2, y2 = scio.loadmat(eval_file)
	x1 = file_contents['x1'].ravel()
	y1 = file_contents['y1'].ravel()
	x2 = file_contents['x2'].ravel()
	y2 = file_contents['y2'].ravel()

	pointsA = np.zeros((len(x1), 2))
	pointsB = np.zeros((len(x2), 2))

	for i in range(len(x1)):
		pointsA[i, 0] = x1[i]
		pointsA[i, 1] = y1[i]
		pointsB[i, 0] = x2[i]
		pointsB[i, 1] = y2[i]

	correct_matches = 0

	F = estimate_fundamental_matrix(ground_truth_correspondence_file)

	for i in range(x1_matches.shape[0]):
		pointA = np.ones((1, 3))
		pointB = np.ones((1, 3))
		pointA[0,0] = x1_matches[i]
		pointA[0,1] = y1_matches[i]
		pointB[0,0] = x2_matches[i]
		pointB[0,1] = y2_matches[i]


		if abs(pointB @ F @ np.transpose(pointA)) < .1:
			x_dists = x1 - x1_matches[i]
			y_dists = y1 - y1_matches[i]

			# computes distances of each feature point to the ground truth point
			dists = np.sqrt(np.power(x_dists, 2.0) + np.power(y_dists, 2.0))
			closest_ground_truth = np.argmin(dists, axis=0)
			offset_x1 = x1_matches[i] - x1[closest_ground_truth]
			offset_y1 = y1_matches[i] - y1[closest_ground_truth]
			offset_x1 *= img_B.shape[0] / img_A.shape[0]
			offset_y1 *= img_B.shape[0] / img_A.shape[0]
			offset_x2 = x2_matches[i] - x2[closest_ground_truth]
			offset_y2 = y2_matches[i] - y2[closest_ground_truth]
			offset_dist = np.sqrt(np.power(offset_x1 - offset_x2, 2) + np.power(offset_y1 - offset_y2, 2))
			if offset_dist < 70:
				correct_matches += 1
				good_matches[i] = True

	accuracy = int(100 * correct_matches / len(matches)) if len(matches) else 0
	print(f'Accuracy on all matches: {accuracy}%')

	print("Vizualizing...")
	visualize.show_correspondences(img_A, img_B, x1_est / scale_factor, y1_est / scale_factor, x2_est / scale_factor, y2_est / scale_factor, matches, good_matches, filename)

	return accuracy


# ---------------------------------------------------------------------------
# DINOv3 Feature Helpers
# ---------------------------------------------------------------------------
#
# DINOv3 is a 2025 method that uses a Vision Transformer (ViT) trained with 
# self-supervised learning to describe images in a 384-d latent space.
# DINOv3 was a multi-million dollar computation investment in GPU time.
# We can compare its feature descriptors to classical SIFT with its 128-d space.
# It runs on your laptop and is derived from first principles.
#
# Unlike SIFT, which computes a descriptor *independently* for each keypoint,
# DINOv3 processes the *entire image at once* and produces a dense grid of
# feature vectors — one 384-d vector per 16×16 pixel patch.
#
# The two public functions below reflect this two-stage process:
#   1. compute_dino_feature_map  — get the dense (H, W, 384) feature grid
#   2. sample_dino_descriptors   — interpolate descriptors at keypoint locations
#
# For the provided image pairs, we load pre-computed feature maps from .npz
# files so no GPU or model download is needed. For custom images, the model
# is downloaded automatically via library timm (~80 MB, first time only).
# ---------------------------------------------------------------------------

def compute_dino_feature_map(image, cache_path=None):
	"""
	Compute (or load from cache) a dense DINOv3 feature map for an image.

	Parameters
	----------
	image : np.ndarray, shape (H, W), dtype float32
		Grayscale image with pixel values in [0, 1].
	cache_path : str or None
		Path to a .npz cache file. If the file exists and matches the image
		dimensions, the cached feature map is returned immediately.

	Returns
	-------
	feature_map : np.ndarray, shape (H_grid, W_grid, 384), dtype float32
		Dense L2-normalized feature vectors on a spatial grid.
		Each grid cell corresponds to one 16×16 pixel patch.
	metadata : dict
		Keys: 'image_h', 'image_w', 'grid_h', 'grid_w', 'patch_size'.
		Needed by sample_dino_descriptors to map pixel coords → grid coords.
	"""
	if cache_path and os.path.exists(cache_path):
		data = np.load(cache_path)
		feature_map = data['feature_map']
		cached_h, cached_w = int(data['image_h']), int(data['image_w'])

		# Validate that the cache was built for the same image size
		if cached_h == image.shape[0] and cached_w == image.shape[1]:
			metadata = {
				'image_h': cached_h,
				'image_w': cached_w,
				'grid_h': feature_map.shape[0],
				'grid_w': feature_map.shape[1],
				'patch_size': 16,
			}
			return feature_map, metadata
		else:
			print("Warning: cached DINOv3 features don't match image size. "
				  "Computing live...")

	return _compute_dino_live(image)


def sample_dino_descriptors(feature_map, metadata, xs, ys):
	"""
	Sample DINOv3 descriptors at keypoint locations via bilinear interpolation.

	This is analogous to get_feature_descriptors() — it returns one descriptor
	per keypoint — but the features come from a pre-trained Vision Transformer
	instead of hand-crafted gradient histograms.

	Parameters
	----------
	feature_map : np.ndarray, shape (H_grid, W_grid, 384)
		Dense feature map from compute_dino_feature_map().
	metadata : dict
		From compute_dino_feature_map().
	xs : np.ndarray, shape (N,)
		X coordinates (column indices) of keypoints in pixel space.
	ys : np.ndarray, shape (N,)
		Y coordinates (row indices) of keypoints in pixel space.

	Returns
	-------
	descriptors : np.ndarray, shape (N, 384), dtype float32
		One L2-normalized descriptor per keypoint.
	"""
	from scipy.ndimage import map_coordinates

	H_img = metadata['image_h']
	W_img = metadata['image_w']
	H_grid = metadata['grid_h']
	W_grid = metadata['grid_w']

	# Map pixel coordinates → continuous grid coordinates
	# The feature grid covers the image uniformly, so we scale linearly.
	gx = np.asarray(xs, dtype=np.float32) * (W_grid / W_img)
	gy = np.asarray(ys, dtype=np.float32) * (H_grid / H_img)

	# Bilinear interpolation (order=1) across all 384 feature channels.
	# mode='nearest' is the *boundary* handling — clamp to edge if out of bounds.
	descriptors = np.zeros((len(xs), feature_map.shape[2]), dtype=np.float32)
	for c in range(feature_map.shape[2]):
		descriptors[:, c] = map_coordinates(
			feature_map[:, :, c], [gy, gx],
			order=1, mode='nearest'
		)

	# L2 normalize each descriptor
	norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
	descriptors = descriptors / np.maximum(norms, 1e-8)

	return descriptors


# --- Private: live model computation (only for custom / uncached images) ---

_DINO_MODEL_CACHE = None

def _compute_dino_live(image):
	"""Run DINOv3 ViT-S/16 on the image. Downloads model on first call (~80 MB)."""
	import timm
	import torch
	import torch.nn.functional as F

	global _DINO_MODEL_CACHE
	PATCH_SIZE = 16

	# Load and cache the model
	if _DINO_MODEL_CACHE is None:
		print("Downloading DINOv3 model (first time only, ~80 MB)...")
		model = timm.create_model(
			'vit_small_patch16_dinov3_qkvb.lvd1689m', pretrained=True
		)
		device = 'cuda' if torch.cuda.is_available() else 'cpu'
		model.eval().to(device)
		_DINO_MODEL_CACHE = (model, device)
	model, device = _DINO_MODEL_CACHE

	# Prepare input: grayscale → RGB, normalize, snap to patch multiples
	H, W = image.shape[:2]
	if image.ndim == 2:
		img_rgb = np.stack([image, image, image], axis=-1)
	else:
		img_rgb = image
	if img_rgb.max() <= 1.0:
		img_rgb = (img_rgb * 255).astype(np.uint8)

	H_r = int(np.ceil(H / PATCH_SIZE) * PATCH_SIZE)
	W_r = int(np.ceil(W / PATCH_SIZE) * PATCH_SIZE)

	x = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0  # (3, H, W)
	x = x.unsqueeze(0)  # (1, 3, H, W)
	x = F.interpolate(x, size=(H_r, W_r), mode='bilinear', align_corners=False)

	# ImageNet normalization (standard for ViT models)
	mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
	std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
	x = (x - mean) / std
	x = x.to(device)

	# Forward pass — extract patch token features
	with torch.inference_mode():
		out = model.forward_features(x)

	# out shape: (1, num_prefix + num_patches, 384)
	# num_prefix includes the CLS token and any register tokens (DINOv3 uses 4).
	# We skip all prefix tokens to get only the spatial patch tokens.
	num_prefix = getattr(model, 'num_prefix_tokens', 1)
	tokens = out[:, num_prefix:, :]  # (1, H_grid * W_grid, 384)

	H_grid = H_r // PATCH_SIZE
	W_grid = W_r // PATCH_SIZE
	feature_map = tokens.reshape(H_grid, W_grid, -1)
	feature_map = F.normalize(feature_map, dim=-1)  # L2 normalize
	feature_map = feature_map.detach().cpu().numpy()

	metadata = {
		'image_h': H, 'image_w': W,
		'grid_h': H_grid, 'grid_w': W_grid,
		'patch_size': PATCH_SIZE,
	}
	return feature_map, metadata
