import os
import sys
import csv
import numpy as np
from PIL import Image

# numbers of picture rows & cols in one 'png' file
RN_PIC, CN_PIC = 3, 3

# the edge pixels of pictures in the 'png' file
R_DIV, C_DIV = [3, 288, 573], [27, 533, 1039]

# the edge length of one pixel of original pictures
PIXEL_SZ = 8

# the size of one original picture
RN_PIXEL, CN_PIXEL = 28, 28

def cut(image):
	'''
	restore original pictures from the png file
	'''
	ret = [[np.zeros((RN_PIXEL, CN_PIXEL)).astype(np.float32) for i in range(RN_PIC)] for j in range(CN_PIC)]
	for i in range(RN_PIC):
		for j in range(CN_PIC):
			for k in range(RN_PIXEL):
				for l in range(CN_PIXEL):
					ret[i][j][k][l] = -(image[R_DIV[i] + k * PIXEL_SZ][C_DIV[j] + l * PIXEL_SZ] - 127.5) / 127.5
	return ret

def ave(matrix):
	'''
	perform the "average" one time
	'''
	ret = np.zeros((RN_PIXEL, CN_PIXEL))
	dx, dy = [-1, 0, 1, 0], [0, 1, 0, -1]
	in_lim = lambda x, y : (0 <= x < RN_PIXEL and 0 <= y < CN_PIXEL)
	for x in range(RN_PIXEL):
		for y in range(CN_PIXEL):
			diff = [np.abs(matrix[x][y] - matrix[x + dx[i]][y + dy[i]]) for i in range(4) if in_lim(x + dx[i], y + dy[i])]
			ret[x][y] = sum(diff) / float(len(diff))
	return ret

def main(path):
	'''
	calculate the blur measurement values of all 'png' files in the current working directory
	'''
	res = []
	for png in [f for f in os.listdir(path) if f[-4:] == '.png']:
		for i in cut(np.asarray(Image.open(os.path.join(path, png)).convert('L'))):
			for j in i:
				res.append(np.sum(ave(ave(j))) / (RN_PIXEL * CN_PIXEL))
	with open('res.csv', 'wb') as f:
		(csv.writer(f)).writerow(res)

main(sys.argv[1])
