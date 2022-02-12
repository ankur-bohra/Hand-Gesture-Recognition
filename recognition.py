import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow import keras

mp_hands = mp.solutions.hands

IMAGE_SIZE = (100, 100)

model = keras.models.load_model('Hand_Gesture_Model.h5')
classes = [chr(num) for num in range(ord('A'), ord('Z')+1)]  # List of all (capital) alphabets
classes.extend(("d", "n", "s"))  # d -> delete; n -> nothing; s -> space

def process_image(image):
	'''Processes the image to be fed into the model.

	Args:
		image: The image to be processed.

	Returns:
		The processed image. 
	'''
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = cv2.Canny(image, 80, 90)  # Edge detection
	image = cv2.dilate(image, None)  # Dilation and erosion
	image = cv2.erode(image, None)   # clears up noise
	image = cv2.resize(image, IMAGE_SIZE)  # Resize to the trained image size
	return image

def get_roi_coords(landmarks, padding=25):
	'''Finds the square region-of-interest coordinates from the hand landmarks.

	Args:
		landmarks: The hand landmarks.
		padding: The padding to be added to the bounding box before translation to roi.
	
	Returns:
		The roi coordinates.
	'''
	for hand_landmarks in results.multi_hand_landmarks:
		x_values = [landmark.x for landmark in hand_landmarks.landmark]
		y_values = [landmark.y for landmark in hand_landmarks.landmark]
		x_min = min(x_values)
		x_max = max(x_values)
		y_min = min(y_values)
		y_max = max(y_values)

	# Scale to actual dimensions
	x_min = int(x_min * width)
	x_max = int(x_max * width)
	y_min = int(y_min * height)
	y_max = int(y_max * height)

	# Apply padding while respecting bounds
    # Minimum values should be decreased during padding, but >= 0
    # Maximum values should be increased but <= image width/height
	x_min = max(x_min - padding, 0)
	x_max = min(x_max + padding, width)
	y_min = max(y_min - padding, 0)
	y_max = min(y_max + padding, height)
	
	# Expand bounding box to square roi
	difference = abs((x_max  - x_min) - (y_max - y_min))
	if x_max - x_min > y_max - y_min:
		# Need to expand the height of the bounding box.
        # Expand the bounding box equally on both sides.
		y_min -= difference // 2
		y_max += difference // 2
		if y_min < 0:
			# The expansion has caused the box to cross the top edge
            # Shift bounding box downwards
			delta = -y_min
			y_min += delta
			y_max += delta
		elif y_max > height:
			# Box has crossed bottom edge, shift bounding box upwards
			delta = y_max-height
			y_min -= delta
			y_max -= delta
	elif x_max - x_min < y_max - y_min:
		# Need to expand the width of the bounding box.
		x_min -= difference // 2
		x_max += difference // 2
		if x_min < 0:
			delta = -x_min
			x_min += delta
			x_max += delta
		elif x_max > width:
			delta = x_max-width
			x_min -= delta
			x_max -= delta
	
	return (x_min, y_min, x_max, y_max)

def predict_class(roi):
	'''Predicts the class of the hand gesture in the image roi.

	Args:
		roi: The region-of-interest image.

	Returns:
		The class of the hand gesture.
	'''
	roi = np.expand_dims(roi, axis=0)  # Add the channel dimension as is in the model
	probabilities = model(roi).numpy()[0]  # Convert tensor result to python object
	prediction = classes[np.argmax(probabilities)]
	return prediction, probabilities

cap = cv2.VideoCapture(0)
_, frame = cap.read()
height, width, channels = frame.shape
with mp_hands.Hands(
		model_complexity=0,  # Less latency with lower complexity
		max_num_hands=1,
		min_detection_confidence=0.5,
		min_tracking_confidence=0.5) as hands:
	while cap.isOpened():
		success, image = cap.read()

		if not success:
			continue

		# To improve performance, optionally mark the image as not writeable to
		# pass by reference.
		image.flags.writeable = False
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # cv2 operates in BGR, mediapipe in RGB
		results = hands.process(image)
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		if results.multi_hand_landmarks:
			# Process roi and predict
			x_min, y_min, x_max, y_max = get_roi_coords(results.multi_hand_landmarks)
			roi = image[y_min:y_max, x_min:x_max]  # Get roi from image
			roi = process_image(roi)
			prediction, probabilities = predict_class(roi)
			
			# Display prediction and roi
			roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
			image[y_min:y_max, x_min:x_max] = cv2.resize(roi, (x_max - x_min, y_max - y_min))  # Paste the roi back into the image
			cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 0), 2)
			image = cv2.flip(image, 1)  # Flip *before* the text is applied
			cv2.putText(image, prediction, (width-x_max, y_max), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1)
		else:
			image = cv2.flip(image, 1)  # Flip for a selfie-view display

		cv2.imshow('Hand Gesture Recognition', image)
		if cv2.waitKey(5)== ord('q'):
			break
cap.release()