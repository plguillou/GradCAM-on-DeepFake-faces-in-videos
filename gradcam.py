from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import cv2
import argparse
import imutils

from classifiers import *

class GradCAM:
	def __init__(self, model, classIdx, layerName=None):
		self.model = model
		self.classIdx = classIdx
		self.layerName = layerName
		if self.layerName is None:
			self.layerName = self.find_layer()

	def find_layer(self):
		for layer in reversed(self.model.layers):
			if len(layer.output_shape) == 4 and not any(x in layer.name for x in ["pool", "batch"]):
				return layer.name
		raise ValueError("Could not find layer, cannot apply CAM.")

	def overlay_heatmap(self, heatmap, image, alpha=0.5,
		colormap=cv2.COLORMAP_JET):
		heatmap = cv2.applyColorMap(heatmap, colormap)
		output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
		return (heatmap, output)

	def compute_heatmap(self, image, eps=1e-8):
		gradModel = Model(
			inputs=[self.model.inputs[0]],
			outputs=[self.model.get_layer(self.layerName).output, self.model.output])
		with tf.GradientTape() as tape:
			inputs = tf.cast(image, tf.float32)
			(convOutputs, predictions) = gradModel(inputs)
			loss = predictions[:, self.classIdx]
		grads = tape.gradient(loss, convOutputs)
		castConvOutputs = tf.cast(convOutputs > 0, "float32")
		castGrads = tf.cast(grads > 0, "float32")
		guidedGrads = castConvOutputs * castGrads * grads
		convOutputs = convOutputs[0]
		guidedGrads = guidedGrads[0]
		weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
		cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)
		(w, h) = (image.shape[2], image.shape[1])
		heatmap = cv2.resize(cam.numpy(), (w, h))
		numer = heatmap - np.min(heatmap)
		denom = (heatmap.max() - heatmap.min()) + eps
		heatmap = numer / denom
		heatmap = (heatmap * 255).astype("uint8")
		return heatmap


def get_heatmap_detection(model, in_imgg):

	orig = in_imgg
	img_rgb = np.copy(orig)
	img_rgb = cv2.resize(img_rgb,(256,256))  # resize
	img_rgb = np.array(img_rgb).astype(np.float32)/255.0  # scaling
	img_rgb = np.expand_dims(img_rgb, axis=0)  # expand dimension

	preds = model.predict(img_rgb)
	i = np.argmax(preds[0])
	# initialize our gradient class activation map and build the heatmap
	cam = GradCAM(model, i)
	heatmap = cam.compute_heatmap(img_rgb)
	# resize the resulting heatmap to the original input image dimensions
	# and then overlay heatmap on top of the image
	heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
	(heatmap, output) = cam.overlay_heatmap(heatmap, orig, alpha=0.5)
	return output

if __name__ == "__main__":

	image_path = "img_test_1.png"

	model = MesoInception4()
	model.load('models/MesoInception_DF.h5')

	img = cv2.imread(image_path)
	img = cv2.resize(img, (300, 300))

	out_img = get_heatmap_detection(model.model, img)

	cv2.imshow("Output", out_img)
	cv2.waitKey(0)