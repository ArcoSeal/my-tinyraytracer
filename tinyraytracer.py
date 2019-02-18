#!/usr/bin/env python3

import numpy as np
from scipy.misc import imshow
from imageio import imwrite

def render():
	width, height = 1024, 768

	frame = np.zeros((height, width, 3)) # [y,x,rgb]

	for rr in range(0, height):
		for cc in range(0, width):
			frame[rr,cc,0] = rr / height
			frame[rr,cc,1] = cc / width
			frame[rr,cc,2] = 0

	return frame

def main():
	frame = render()
	imwrite('./test.png', frame)
	imshow(frame)

if __name__ == '__main__':
	main()