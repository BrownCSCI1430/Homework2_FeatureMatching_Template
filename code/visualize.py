
from skimage.feature import plot_matched_features
import matplotlib
#matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import os

def show_correspondences(imgA, imgB, X1, Y1, X2, Y2, matches, good_matches, filename=None, match_errors=None):
	'''
		Visualizes corresponding points between two images, either as
		arrows or dots

		mode='dots': Corresponding points will have the same random color
		mode='arrows': Corresponding points will be joined by a line

		Writes out a png of the visualization if 'filename' is not None.

		If match_errors is provided, an AUC accuracy curve is plotted below
		the correspondence image.
	'''

	if match_errors is not None:
		fig, (ax, ax_auc) = plt.subplots(nrows=1, ncols=2,
			gridspec_kw={'width_ratios': [3, 1]}, figsize=(14, 5))
	else:
		fig, ax = plt.subplots(nrows=1, ncols=1)

	kp1 = zip_x_y(Y1, X1)
	kp2 = zip_x_y(Y2, X2)
	matches = matches.astype(int)
	plot_matched_features(imgA, imgB,
		keypoints0=kp1, keypoints1=kp2,
		matches=matches[np.logical_not(good_matches)],
		ax=ax, matches_color='orangered')
	plot_matched_features(imgA, imgB,
		keypoints0=kp1, keypoints1=kp2,
		matches=matches[good_matches],
		ax=ax, matches_color='springgreen')

	n_good = int(good_matches.sum())
	n_total = len(good_matches)
	n_bad = n_total - n_good
	ax.set_title(f'Matches: {n_good} good / {n_bad} bad / {n_total} total', fontsize=10)

	if match_errors is not None:
		max_threshold = 100
		thresholds = np.arange(1, max_threshold + 1)
		accuracies = np.array([(match_errors < t).mean() * 100 for t in thresholds])
		auc_score = int(np.round(accuracies.mean()))

		ax_auc.fill_between(thresholds, accuracies, alpha=0.25, color='steelblue')
		ax_auc.plot(thresholds, accuracies, color='steelblue', linewidth=2)

		# Mark specific thresholds
		markers = [3, 5, 10, 25, 50]
		marker_accs = [(match_errors < t).mean() * 100 for t in markers]
		ax_auc.scatter(markers, marker_accs, color='steelblue', zorder=5, s=40)
		for t, a in zip(markers, marker_accs):
			ax_auc.annotate(f'{a:.0f}%', (t, a), textcoords='offset points',
				xytext=(5, 5), fontsize=8)

		ax_auc.set_xlabel('Distance threshold (pixels)')
		ax_auc.set_ylabel('Accuracy (%)')
		ax_auc.set_title(f'Area Under Curve (AUC): {auc_score}%')
		ax_auc.set_xlim(0, max_threshold)
		ax_auc.set_ylim(0, 105)
		ax_auc.grid(True, alpha=0.3)

	fig.tight_layout()
	fig = plt.gcf()
	plt.show()

	if filename:
		if not os.path.isdir('../results'):
			os.mkdir('../results')
		fig.savefig('../results/' + filename)

	return

def show_correspondences_custom_image(imgA, imgB, X1, Y1, X2, Y2, matches, scale_factor, filename=None):
	'''
		Visualizes corresponding points between two images, either as
		arrows or dots. Unlike show_correspondences, does not take correct_matches argument

		mode='dots': Corresponding points will have the same random color
		mode='arrows': Corresponding points will be joined by a line

		Writes out a png of the visualization if 'filename' is not None.
	'''

	# generates unique figures so students can
	# look at all three at once
	fig, ax = plt.subplots(nrows=1, ncols=1)

	x1_scaled = X1 / scale_factor
	y1_scaled = Y1 / scale_factor
	x2_scaled = X2 / scale_factor
	y2_scaled = Y2 / scale_factor

	kp1 = zip_x_y(y1_scaled, x1_scaled)
	kp2 = zip_x_y(y2_scaled, x2_scaled)
	matches = matches.astype(int)
	plot_matched_features(imgA, imgB,
		keypoints0=kp1, keypoints1=kp2,
		matches=matches,
		ax=ax, matches_color='yellow')

	fig = plt.gcf()
	plt.show()

	if filename:
		if not os.path.isdir('../results'):
			os.mkdir('../results')
		fig.savefig('../results/' + filename)

	return

def zip_x_y(x, y):
	zipped_points = []
	for i in range(len(x)):
		zipped_points.append(np.array([x[i], y[i]]))
	return np.array(zipped_points)
