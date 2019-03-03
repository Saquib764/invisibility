import cv2
import numpy as np
import copy
import os
import time
from sys import exit
import scipy
import imutils

kernel = np.ones((5,5),np.uint8)
kernel_close = np.ones((9,9),np.uint8)


# define range of blue color in HSV
lower_blue = np.array([110,20,2])
upper_blue = np.array([140,130,140])

# define range of Green color in HSV
lower_blue = np.array([80,80,10])
upper_blue = np.array([100,255,220])

hsv = None
# Pre-process
def constrastLimit(image):
	img_hist_equalized = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
	channels = cv2.split(img_hist_equalized)
	channels[0] = cv2.equalizeHist(channels[0])
	img_hist_equalized = cv2.merge(channels)
	img_hist_equalized = cv2.cvtColor(img_hist_equalized, cv2.COLOR_YCrCb2BGR)
	return img_hist_equalized

def LaplacianOfGaussian(image):
	LoG_image = cv2.GaussianBlur(image, (3,3), 0)           # paramter 
	image = cv2.cvtColor(LoG_image, cv2.COLOR_BGR2HSV)
	# gray = cv2.cvtColor( LoG_image, cv2.COLOR_BGR2GRAY)
	gray = image[:,:,2]
	LoG_image = cv2.Laplacian( gray, cv2.CV_8U,3,3,3)       # parameter
	LoG_image = cv2.convertScaleAbs(LoG_image)
	return LoG_image

def Sobels(image):
	LoG_image = cv2.GaussianBlur(image, (7,7), 0)           # paramter 
	image = cv2.cvtColor(LoG_image, cv2.COLOR_BGR2HSV)
	# gray = cv2.cvtColor( LoG_image, cv2.COLOR_BGR2GRAY)
	gray = image[:,:,2]
	sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=5)
	sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=5)

	return sobelx, sobely
	
def binarization(image):
	global hsv
	image = cv2.GaussianBlur(image, (7,7), 0)   
	image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

	hsv = copy.copy(image)
	# cv2.imshow("Video", image)

	mask = cv2.inRange(image, lower_blue, upper_blue)

	# cv2.imshow("Mask", mask)

	return mask

def preprocess_image(image):
	image = constrastLimit(image)
	# image = LaplacianOfGaussian(image)
	image = binarization(image)
	return image

def removeSmallComponents(image, threshold):
    #find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    img2 = np.zeros((output.shape),dtype = np.uint8)
    #for every component in the image, you keep it only if it's above threshold
    for i in range(0, nb_components):
        if sizes[i] >= threshold:
            img2[output == i + 1] = 255
    return img2


def mouse_action(event, x, y, flag, param):
	if event == cv2.EVENT_LBUTTONDOWN:
		print(hsv[y,x])

# cap = cv2.VideoCapture("video/MVI_2619.MOV")
cap = cv2.VideoCapture("mobile_towel.mp4")
# cap = cv2.VideoCapture(0)



# cv2.namedWindow("Video", flags = cv2.WINDOW_AUTOSIZE)
# cv2.setMouseCallback("Video", mouse_action)



def read(capt):
	ret, frame = capt.read()
	if not ret:
		return ret, None
	# frame = cv2.resize(frame,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
	H,W, __ = frame.shape
	frame = imutils.rotate_bound(frame, -90)
	# M = cv2.getRotationMatrix2D((W/2,H/2),90,1)
	# frame = cv2.warpAffine(frame,M,(W,H))
	return ret, frame

for i in range(30*9):
	ret, frame = read(cap)

ret, frame = read(cap)
bg = copy.copy(frame)

H,W, __ = frame.shape


fps = cap.get(cv2.CAP_PROP_FPS)
print fps
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, fps , (W,H))


print frame.shape

ri = np.array([np.arange(H)]).T
ri = np.repeat(ri, [W], axis=1)


ci = np.array([np.arange(W)])
ci = np.repeat(ci, [H], axis=0)

# cv2.imshow("Yo", frame)
# cmd = cv2.waitKey(0)

# exit()
while(True):
	# Read frame
	# for i in range(3):
	# 	ret, frame = read(cap)
	
	ret, frame = read(cap)
	if not ret:
		break;

	frame_original = copy.copy(frame)

	mask = preprocess_image(frame)

	mask2 = removeSmallComponents(mask, 1000)
	mask2 = mask2.reshape((H, W,1))
	mask2 = cv2.dilate(mask2, kernel,iterations = 2)
	mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel_close)
	mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel_close)


	bg_patch = cv2.bitwise_and(bg,bg,mask = mask2)
	# bg_patch = np.multiply(bg, mask2)



	image_hole = cv2.bitwise_and(frame_original,frame_original,mask = ~mask2)


	removed_patch = cv2.bitwise_and(frame_original,frame_original,mask = mask2)
	# lap_patch = LaplacianOfGaussian(removed_patch)

	A = 10
	sx, sy = np.clip(np.int32(A*np.int16(Sobels(removed_patch))/1048), -A, A)

	y = np.clip(ri + sy, 0, H-1)
	x = np.clip(ci + sx, 0, W-1)

	# distorted_bg_patch = copy.copy(bg_patch)
	distorted_bg_patch = bg[y, x]


	distorted_bg_patch = cv2.bitwise_and(distorted_bg_patch,distorted_bg_patch,mask = mask2)
	replaced = image_hole + distorted_bg_patch

	# replaced = cv2.flip( replaced, 1 )

	# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# cv2.imshow("Video", frame)
	# cv2.imshow("mask", mask)
	# cv2.imshow("Out", replaced)
	# cv2.imshow("Outp", distorted)

	out.write(replaced)


	# cmd = cv2.waitKey(0)
	# # if cmd > 0:
	# # print cmd
	# if cmd == 27:
	# 	break

cap.release()
out.release()
cv2.destroyAllWindows()
