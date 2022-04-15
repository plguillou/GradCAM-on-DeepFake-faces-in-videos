from imutils.video import VideoStream
from imutils import face_utils
import imutils
import dlib
import cv2

from classifiers import *
import numpy as np

import datetime
import time

from gradcam import get_heatmap_detection


class VideoGanAnalyzer:

	def __init__(self, name_of_video, model_path, is_gan_threshold=0.5):
		self.name_of_video = name_of_video
		self.model_path = model_path
		self.is_gan_threshold = is_gan_threshold
		self.detector = None
		self.vs = None
		self.fps = None
		self.writer = None
		self.H = None
		self.W = None
		self.model_input_shape = (256, 256)
		self.list_frames = []
		self.list_of_faces = []
		self.classifier = None

	def load_face_detector(self):
		print("[INFO] loading frontal face detector...")
		self.detector = dlib.get_frontal_face_detector()

	def load_video(self):
		print("[INFO] loading video...")
		self.vs = cv2.VideoCapture(self.name_of_video)
		self.fps = self.vs.get(cv2.CAP_PROP_FPS)
		print("[INFO] fps : {0}".format(self.fps))
		time.sleep(2.0)

	def add_face_to_list(self, face_image, id_frame, id_face, coord):
		"""{"image" : np.array, "id_frame" : int, "coord" : [x, y, w, h]}"""
		return {"image" : face_image, "image_resized" : cv2.resize(np.copy(face_image), self.model_input_shape),
				"id_frame" : id_frame, "id_face" : id_face, "coord" : coord}

	def read_video(self):
		print("[INFO] reading the video...")
		while True:
			(grabbed, frame) = self.vs.read()
			if not grabbed:
				break
			self.list_frames.append(frame)
		assert 0 < len(self.list_frames), "[WARNING] video empty, check the file name..."

	def detect_faces(self):
		print("[INFO] detecting faces...")
		for id_frame, frame in enumerate(self.list_frames):

			if self.W is None or self.H is None:
				(self.H, self.W) = frame.shape[:2]
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

			rects = self.detector(gray, 0)

			# for each face
			for (i, rect) in enumerate(rects):
				# convert dlib's rectangle to a OpenCV-style bounding box
				# [i.e., (x, y, w, h)], then draw the face bounding box
				(x, y, w, h) = face_utils.rect_to_bb(rect)
				# upsizing the rectangle
				modif_factor = 0.2
				modif_w = int(modif_factor * w)
				x -= modif_w
				w += 2*modif_w
				modif_h = int(modif_factor * h)
				y -= modif_h
				h += 2*modif_h

				self.list_of_faces.append(self.add_face_to_list(frame[y:y+h, x:x+w], id_frame, i, [x, y, w, h]))

	def load_classifier(self):
		# Load the model and its pretrained weights
		print("[INFO] loading the model...")
		self.classifier = MesoInception4()
		self.classifier.load(self.model_path)

	def gan_analysis(self):
		print("[INFO] analyzing the video...")
		for face in self.list_of_faces:

			frame = self.list_frames[face["id_frame"]]
			x, y, w, h = face["coord"]
			i = face["id_face"]

			pred = self.classifier.predict(np.expand_dims(face["image_resized"], axis=0)/255.)
			pred = float(pred)

			gan_detected = False
			if pred < self.is_gan_threshold:
				gan_detected = True

			# BGR instead of RGB because it is cv2
			color_is_gan = (0, 255, 0) if not gan_detected else (0, 0, 255)

			if gan_detected:
				frame[y:y+h, x:x+w] = get_heatmap_detection(self.classifier.model, face["image"])

			cv2.rectangle(frame, (x, y), (x + w, y + h), color_is_gan, 3)
			cv2.putText(frame, "Face #{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1., color_is_gan, 3)

	def save_video(self):
		print("[INFO] saving the video...")
		for frame in self.list_frames:
			# check if the video writer is None
			if self.writer is None:
				# initialize our video writer
				fourcc = cv2.VideoWriter_fourcc(*"MJPG")
				self.writer = cv2.VideoWriter(self.name_of_video[:-4] + "_analyzed.avi", fourcc, self.fps, (self.W, self.H), True)
			# write the output frame to disk
			self.writer.write(frame)

	def clean_end(self):
		# release the file pointers
		print("[INFO] cleaning up...")
		self.writer.release()
		self.vs.release()

	def main(self):
		self.load_face_detector()
		self.load_video()
		self.read_video()
		self.detect_faces()
		self.load_classifier()
		self.gan_analysis()
		self.save_video()
		self.clean_end()


if __name__ == "__main__":

	name_of_video = "vid8.mp4"
	model_path = "models/MesoInception_DF.h5"

	analyzer = VideoGanAnalyzer(name_of_video, model_path)
	analyzer.main()

