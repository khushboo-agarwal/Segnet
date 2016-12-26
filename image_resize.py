import os
import cv2

path = '/Users/administrator/PDFS/Segnet/CamVid2'
imlist = []

im = os.listdir(path)
for i in im:
	if i != '.DS_Store':
		im1 = os.path.join(path, i)
		im2 = os.listdir(im1)
		for j in im2:
			if j != '.DS_Store':
				im3 = os.path.join(im1, j)
				if (im3 != '.DS_Store') and (im3.endswith('.png')):
					imlist.append(im3)
for i in imlist:
	I = cv2.imread(i)
	H, W = I.shape[:2]
	resize = cv2.resize(I, (224, 224))
	cv2.imwrite(i, resize)
	# aspect = W/(H*1.0)

