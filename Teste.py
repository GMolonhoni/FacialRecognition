from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from scipy import misc
import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
import facenet
import detect_face
import os
from os.path import join as pjoin
import sys
import time
import copy
import math
import pickle
from sklearn.svm import SVC
from sklearn.externals import joblib

print("Program Starting")

####### Global Variables ###
minsize = 0
pnet = None
rnet = None
onet = None
threshold = None
factor = 0
sess = None
model = None
names = []
images_placeholder = None
embedding_size = None
image_size = 182
input_image_size = 160
phase_train_placeholder = None
embeddings = None

####### Functions ##########

def Setup():
	global minsize, sess, pnet, rnet, onet, threshold, factor, names, images_placeholder, embedding_size, image_size, input_image_size
	global phase_train_placeholder, embeddings
	
	
	print("Setup...")
	with tf.Graph().as_default():
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
		sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
		with sess.as_default():
			pnet, rnet, onet = detect_face.create_mtcnn(sess, './d_npy')

			minsize = 20  # minimum size of face
			threshold = [0.6, 0.7, 0.7]  # three steps's threshold
			factor = 0.709  # scale factor
			margin = 44
			batch_size = 1000

        
			names = os.listdir("./Raw_Dataset")
			names.sort()
			
			Load_Feature_Exctractor()
		
			images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
			embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
			phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
			embedding_size = embeddings.get_shape()[1]
		
			
			Load_Classifier()
			
		
		
def Load_Feature_Exctractor():
	print('Loading feature extraction model')
	modeldir = './pre_model/20170512-110547.pb'
	facenet.load_model(modeldir)
	print('Feature extraction model loaded')
	
def Load_Classifier():
	global model
	classifier_filename = './Classifier/my_classifier.pkl'
	classifier_filename_exp = os.path.expanduser(classifier_filename)
	with open(classifier_filename_exp, 'rb') as infile:
		(model, class_names) = pickle.load(infile)
	print('loaded classifier file-> %s' % classifier_filename_exp)
	
def Open_Camera():
	print('opening camera')
	video_capture = cv2.VideoCapture(0)
	return video_capture
	# #video writer
	
	
	
def Face_Recognize(frame):
	global minsize, pnet, rnet, onet,threshold, factor, sess, embedding_size, image_size, phase_train_placeholder, embeddings, embeddings

	if frame.ndim == 2:
		frame = facenet.to_rgb(frame)
	frame = frame[:, :, 0:3]
	bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
	nrof_faces = bounding_boxes.shape[0]
				
	print('Detected Faces: %d' % nrof_faces)

	if nrof_faces > 0:
		det = bounding_boxes[:, 0:4]
		img_size = np.asarray(frame.shape)[0:2]

		cropped = []
		scaled = []
		scaled_reshape = []
		bb = np.zeros((nrof_faces,4), dtype=np.int32)

		for i in range(nrof_faces):
			emb_array = np.zeros((1, embedding_size))
			bb[i][0] = det[i][0]
			bb[i][1] = det[i][1]
			bb[i][2] = det[i][2]
			bb[i][3] = det[i][3]

                        # inner exception
			if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
				print('face is inner of range!')
				continue

			cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
			cropped[i] = facenet.flip(cropped[i], False)
			scaled.append(misc.imresize(cropped[i], (image_size, image_size), interp='bilinear'))
			scaled[i] = cv2.resize(scaled[i], (input_image_size,input_image_size),
			interpolation=cv2.INTER_CUBIC)
			scaled[i] = facenet.prewhiten(scaled[i])
			scaled_reshape.append(scaled[i].reshape(-1,input_image_size,input_image_size,3))
			feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
			emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
			predictions = model.predict_proba(emb_array)
			print("Distances:" )
			print(predictions)
			best_class_indices = np.argmax(predictions, axis=1)
			best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
			cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)    #boxing face

			#plot result idx under box
			text_x = bb[i][0]
			text_y = bb[i][3] + 20
			print('Names: ')
			print(names)
			for H_i in names:
				if names[best_class_indices[0]] == H_i:
					result_names = names[best_class_indices[0]]
					print("Person: " + result_names)
					cv2.putText(frame, result_names, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
						1, (0, 0, 255), thickness=1, lineType=2)
	else:
		print('Unable to align')
		
	cv2.imshow('Video', frame)
	
def Loop():
	cam = Open_Camera()
	while True:
		ret, frame = cam.read()
		Face_Recognize(frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cam.release()
	cv2.destroyAllWindows()
	
	
	
	
Setup()
Loop()
#if __name__ == "__main__":
	