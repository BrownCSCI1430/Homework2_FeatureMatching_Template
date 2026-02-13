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

def evaluate_correspondence(img_A, img_B, ground_truth_correspondence_file,
	scale_factor, x1_est, y1_est, x2_est, y2_est, matches, filename="notre_dame_matches.jpg"):

	if len(matches) == 0:
		print('No matches to evaluate.')
		return 0

	# Unscale student coordinates to original image space
	x1_est_scaled = x1_est / scale_factor
	y1_est_scaled = y1_est / scale_factor
	x2_est_scaled = x2_est / scale_factor
	y2_est_scaled = y2_est / scale_factor

	# Extract matched point coordinates
	idx1 = matches[:, 0].astype(int)
	idx2 = matches[:, 1].astype(int)
	pts1_m = np.column_stack([x1_est_scaled[idx1], y1_est_scaled[idx1]])  # (M, 2)
	pts2_m = np.column_stack([x2_est_scaled[idx2], y2_est_scaled[idx2]])  # (M, 2)

	# Load ground truth correspondences
	file_contents = scio.loadmat(ground_truth_correspondence_file)
	x1_gt = file_contents['x1'].ravel()
	y1_gt = file_contents['y1'].ravel()
	x2_gt = file_contents['x2'].ravel()
	y2_gt = file_contents['y2'].ravel()
	pts1_gt = np.column_stack([x1_gt, y1_gt])  # (G, 2)
	pts2_gt = np.column_stack([x2_gt, y2_gt])  # (G, 2)

	# Compute symmetric transfer error for each student match against all GT pairs.
	# For match i and GT pair j: error = max(||p1_est_i - p1_gt_j||, ||p2_est_i - p2_gt_j||)
	# Then match_error_i = min_j(error_ij)
	d1 = np.sqrt(((pts1_m[:, None, :] - pts1_gt[None, :, :]) ** 2).sum(axis=2))  # (M, G)
	d2 = np.sqrt(((pts2_m[:, None, :] - pts2_gt[None, :, :]) ** 2).sum(axis=2))  # (M, G)
	match_errors = np.maximum(d1, d2).min(axis=1)  # (M,)

	# AUC across thresholds 1..100 pixels
	max_threshold = 100
	thresholds = np.arange(1, max_threshold + 1)
	accuracies = np.array([(match_errors < t).mean() for t in thresholds])
	auc_score = int(np.round(accuracies.mean() * 100))

	# Good matches for visualization (correct within 50 pixels)
	good_matches = match_errors < 50

	# Print accuracy at distance thresholds
	print(f'Accuracy at distance thresholds ({len(matches)} matches):')
	print(f'(acc@Xpx = % of matches within X pixels of a ground truth match)')
	for t in [3, 5, 10, 25, 50]:
		acc_at_t = (match_errors < t).mean() * 100
		print(f'  acc@{t}px: {acc_at_t:.1f}%')
	print(f'  Median error: {np.median(match_errors):.1f}px')
	print(f'  Area under curve (AUC): {auc_score}%')
	print(f'  (AUC averages accuracy across all thresholds from 1 to 100px.'
		  f' A perfect score is 100%.)')

	print("Visualizing...")
	visualize.show_correspondences(img_A, img_B, x1_est / scale_factor, y1_est / scale_factor, x2_est / scale_factor, y2_est / scale_factor, matches, good_matches, filename, match_errors)

	return auc_score


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
