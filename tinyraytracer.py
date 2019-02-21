#!/usr/bin/env python3

import sys

import numpy as np
from imageio import imwrite
from matplotlib.pyplot import imshow

class Material():
    def __init__(self, colour, diffuse_albedo, specular_albedo, specular_exp):
        self.colour = np.array(colour)
        self.diffuse_albedo = diffuse_albedo
        self.specular_albedo = specular_albedo
        self.specular_exp = specular_exp

class Ray():
    def __init__(self, origin, direction):
        self.origin = np.array(origin)
        self.direction = normalise(np.array(direction))

class Sphere():
    def __init__(self, centre, radius, material):
        self.centre = np.array(centre)
        self.radius = np.array(radius)
        self.material = material

    def ray_intersect(self, ray):
        # TODO: try out geometric solution too
        ac = ray.origin - self.centre
        a = np.dot(ray.direction, ray.direction)
        b = 2 * np.dot(ray.direction, ac)
        c = np.dot(ac, ac) - self.radius**2

        discriminant = b**2 - 4*a*c

        if discriminant >= 0:
            t = (-1*b - discriminant**0.5) / (2*a) # magnitude of vector from ray origin to intersection (smaller solution of quadratic is always closest to origin)
            return t
        else:
            return None

    def normal(self, point):
        return normalise(point - self.centre)

class Light():
    def __init__(self, position, intensity):
        self.position = np.array(position)
        self.intensity = intensity

def normalise(vector):
    return vector / np.linalg.norm(vector)

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

def do_lighting(ray, lights, point, sphere):
    normal = sphere.normal(point)
    material = sphere.material

    diffuse_light_intensity, specular_light_intensity = 0, 0
    for light in lights:
        light_dir = normalise(light.position - point)

        # diffuse 
        diffuse_light_intensity += light.intensity * max(0, np.dot(light_dir, normal)) 

        # specular
        reflection_dir = normalise(2*np.dot(light_dir, normal)*normal - light_dir)
        specular_light_intensity += light.intensity * max(0, np.dot(reflection_dir, -1*ray.direction)) ** material.specular_exp

    diffuse_colour = material.diffuse_albedo * diffuse_light_intensity * material.colour
    specular_colour = material.specular_albedo * specular_light_intensity * WHITE_COLOUR
    colour = diffuse_colour + specular_colour

    return colour

def cast_ray(ray, spheres, lights):
    # TODO tidy this up
    min_t = float('inf')
    nearest_sphere = None

    for sphere in spheres:
        t = sphere.ray_intersect(ray)
        if t is not None and t < min_t:
            min_t = t
            nearest_sphere = sphere

    if nearest_sphere is not None:
        intersection = ray.origin + min_t * ray.direction
        colour = do_lighting(ray, lights, intersection, nearest_sphere)
        return colour
        
    else:
        return BG_COLOUR

def render(spheres, lights):
    width, height = 1024, 768
    camera_pos = [0,0,0]

    frame = np.zeros((height, width, 3)) # [y,x,rgb]

    for y_px in range(0, height):
        for x_px in range(0, width):
            print('Rendering pixel ({},{})...\r'.format(x_px, y_px), end='')

            x_r, y_r = px2coords((x_px, y_px), (width, height), 1, 90)
            ray_dir = normalise([x_r, y_r, -1])
            ray = Ray(camera_pos, ray_dir)

            frame[y_px,x_px,:] = cast_ray(ray, spheres, lights)

    print('')

    return frame

def main():
    spheres = [Sphere([-3.0,  0.0, -16.0], 2, MATERIALS['ivory']),
                Sphere([-1.0, -1.5, -12.0], 2, MATERIALS['red_rubber']),
                Sphere([ 1.5, -0.5, -18.0], 3, MATERIALS['red_rubber']),
                Sphere([ 7.0,  5.0, -18.0], 4, MATERIALS['ivory'])]

    lights = [Light([-20, 20,  20], 1.5),
                Light([30, 50, -25], 1.8),
                Light([30, 20,  30], 1.7)]

    frame = render(spheres, lights)
    
    imwrite('./test.png', frame)
    imshow(frame)

BG_COLOUR = np.array([0.2, 0.7, 0.8])
WHITE_COLOUR = np.array([1.0, 1.0, 1.0])

MATERIALS = {
            'ivory':        Material(colour=(0.4, 0.4, 0.3), diffuse_albedo=0.6, specular_albedo=0.3, specular_exp=50),
            'red_rubber':   Material(colour=(0.3, 0.1, 0.1), diffuse_albedo=0.9, specular_albedo=0.1, specular_exp=10)
            }

if __name__ == '__main__':
    main()