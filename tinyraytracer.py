#!/usr/bin/env python3

import numpy as np
from scipy.misc import imshow
from imageio import imwrite

class Ray():
	def __init__(self, origin, direction):
		self.origin = origin
		self.direction = normalise(direction)

class Sphere():
	def __init__(self, centre, radius):
		self.centre = centre
		self.radius = radius

	def ray_intersect(self, ray):
		# TODO: try out geometric solution too
		ac = ray.origin - self.centre
		a = np.dot(ray.direction, ray.direction)
		b = 2 * np.dot(ray.direction, ac)
		c = np.dot(ac, ac) - self.radius**2

		discriminant = b**2 - 4*a*c

		if discriminant >= 0:
			return True
		else:
			return False

def normalise(vector):
	return vector / np.linalg.norm(vector)

def cast_ray(ray, sphere):
	if sphere.ray_intersect(ray):
		return SPHERE_COLOUR
	else:
		return BG_COLOUR

def px2coords(px_coords, screen_size, z_dist, fov_deg):
	# TODO: optimise
	x_px, y_px = px_coords
	w_px, h_px = screen_size
	aspect = w_px/h_px
	fov_rad = np.deg2rad(fov_deg)

	x_px_c = (2*x_px - w_px - 1) / 2 # x posn of pixel centre relative to screen centre in px
	w_r = 2 * z_dist * np.tan(fov_rad/2) # width of screen in real coords
	x_r = x_px_c / (w_px/2) * (w_r/2) # x posn of pixel centre in real coords
	x_r *= aspect # correct for aspect ratio

	y_px_c = (2*y_px - h_px - 1) / 2 # y posn of pixel centre relative to screen centre in px
	h_r = 2 * z_dist * np.tan(fov_rad/2) # height of screen in real coords
	y_r = y_px_c / (h_px/2) * (h_r/2) # y posn of pixel centre in real coords
	y_r *= -1 # correct direction (increasing y px coords -> downwards)

	return (x_r, y_r)

def render(sphere):
	width, height = 1024, 768
	camera_pos = np.array([0,0,0])

	frame = np.zeros((height, width, 3)) # [y,x,rgb]

	for y_px in range(0, height):
		for x_px in range(0, width):
			print('Rendering pixel ({},{})...\r'.format(x_px, y_px), end='')

			x_r, y_r = px2coords((x_px, y_px), (width, height), 1, 90)
			ray_dir = normalise(np.array([x_r, y_r, -1]))
			ray = Ray(camera_pos, ray_dir)
			frame[y_px,x_px,:] = cast_ray(ray, sphere)

	print('')

	return frame

def main():
	sphere1 = Sphere(np.array([-3, 0, -16]), 2)
	frame = render(sphere1)
	
	imwrite('./test.png', frame)
	imshow(frame)

BG_COLOUR = (0.2, 0.7, 0.8)
SPHERE_COLOUR = (0.4, 0.4, 0.3)

if __name__ == '__main__':
	main()