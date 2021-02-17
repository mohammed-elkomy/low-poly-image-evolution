import os
import pickle
from functools import partial
from multiprocessing import Pool
from random import randrange

import numpy as np
from tqdm import tqdm

from bezier import get_bezier_curve
import cv2

#####################################################################################
############################ Fancy Animated video demo ##############################
#####################################################################################

MAX_COLOR = 255

STEER = 0.2
SMOOTHNESS = 0.05
NUM_ANCHORS = 5
ANIMATION_STEPS = 60  # must be divisible by (NUM_ANCHORS-1)

LAYERS = 250  # we can 16 polygons in parallel
OVERLAP_PERCENTAGE = .9


def sample_point_L_shaped(upper_left, lower_right, size=(2,)):
    """
    I would like to sample the curve anchor points from the X points as shown below, dividing this into three rectangles
    ##### XXX
    ##### XXX
    ##### XXX
    ##### XXX
    XXXXX|XXX|
    XXXXX|XXX|
    :param upper_left: the upper left of the outer rectangle (surrounded by |)
    :param lower_right: the lower right of the outer rectangle (surrounded by |), higher than upper_left point
    :return: sampled point in the L shape of X's
    """
    if not np.alltrue(upper_left == [0, 0]):  # if upper_left is origin don't do the L
        portion = randrange(0, 3)  # we have three portions of XXX, upper | left | lower right

        if portion == 0:  # left
            lower_right[0] = upper_left[0]
            upper_left[0] = 0
        elif portion == 1:  # upper
            lower_right[1] = upper_left[1]
            upper_left[1] = 0

    return np.random.randint(upper_left, lower_right, size=size)


def get_random_path(draw_object, shape, num_steps):
    """
    for a given draw shape get an appealing path, (doesn't allow very short paths)
    :param draw_object: the object to animate
    :param shape: the draw window shape
    :param num_steps: resolution of the curve
    :return: the random path obtained
    """
    shape = shape[:2]  # width + height
    shape = shape[::-1]
    shape = np.array(shape)
    path_points = np.zeros((NUM_ANCHORS, 2), dtype=np.int)  # a path has 3 points (1 outside, 1 inside, and the origin)
    num_segments = (NUM_ANCHORS - 1)
    weights = np.linspace(0, 1.25, num_segments + 1)
    draw_object_center = draw_object["points"].mean(axis=0)
    is_negative = (shape - draw_object_center) < draw_object_center

    for i in range(num_segments):
        upper_weight = weights[-i - 1]
        lower_weight = weights[-i - 2]
        path_points[i] = sample_point_L_shaped(shape * lower_weight, shape * upper_weight, size=(2,))

    if is_negative[0]:
        path_points[..., 0] *= -1  # the draw object is on the right, reflect the draw path to start from left
    if is_negative[1]:
        path_points[..., 1] *= -1  # the draw object is at the bottom of the canvas, reflect the draw path to start from above

    x, y, _ = get_bezier_curve(path_points, STEER=STEER, smooth=SMOOTHNESS, numpoints=num_steps // num_segments)
    return x, y


def shifted_draw(draw_objects, path_shift, canvas_shape, ):
    """
    draw the draw_objects and return the image
    :param draw_objects: the objects to draw
    :param path_shift: the amount of shift for every object, a nan means don't draw
    :param canvas_shape: the draw window size
    :return: the image drawn as a numpy array
    """
    image = np.zeros(canvas_shape, dtype=np.uint8)
    # shuffle(draw_objects) # don't shuffle the polygons :), they are overlaid over each other
    for idx, draw_object in enumerate(draw_objects):
        if not np.isnan(path_shift[idx, 0]):
            polygon = draw_object["points"]
            color = draw_object["color"]
            polygon = polygon.astype(np.int) + path_shift[idx].astype(np.int)
            color = color.astype(np.int)
            overlay = image.copy()
            color = color.tolist()
            BGR, alpha = color[:-1], color[-1] / MAX_COLOR

            cv2.drawContours(overlay, [polygon], 0, BGR, -1)
            # apply the overlay
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    return image


# the worker for multi-processing, every worker is responsible for drawing the image for a certain animation step
def worker(step, draw_objects, draw_objects_shift_layers, canvas_shape, video_frames_dist
           ):
    """
    draws the image for a given animation step, and saves it to the disk
    ################################### SHARED ACROSS PROCESSES ##########################
    # that's why I will use partial to set them before sending them to the pool
    #####################################################################################
    :param video_frames_dist: destination path for video frames
    :param canvas_shape:  the size of the canvas
    :param draw_objects_shift_layers: the shift for each polygon for a given step in time
    :param draw_objects: the polygons to draw
    #####################################################################################
    :param step: the animation step
    """
    image = shifted_draw(draw_objects, draw_objects_shift_layers[..., step], canvas_shape)

    cv2.imwrite(os.path.join(video_frames_dist, "{:05d}.png".format(step)), image)


def render_video_frames(video_frames_dist, source_draw_objects_dict):
    with open(source_draw_objects_dict, 'rb') as f:
        draw_objects = pickle.load(f)
        canvas_shape = (0, 0)  # automatically find the shape through maximizing
        for draw_object in draw_objects:
            points = draw_object["points"]
            canvas_shape = np.maximum(canvas_shape, points.max(axis=0))

        # generate random paths for each draw object
        canvas_shape = canvas_shape.astype(np.int)
        canvas_shape = tuple(canvas_shape[::-1]) + (3,)
        draw_objects = draw_objects
        draw_objects_shift = np.array([get_random_path(draw_object, canvas_shape, ANIMATION_STEPS) for draw_object in draw_objects], dtype=np.float64)

        # generate the overlapping shift array, it's done through splitting the polygons into layers
        # the polygons on the lower layers are rendered before the upper layers
        # the number of layers is the number of polygons per sub-image in the optimization script
        shift_per_layer = int(np.ceil(ANIMATION_STEPS * (1 - OVERLAP_PERCENTAGE)))  # the shift between layers, due to overlapping/interleaving
        total_animation_steps = ((LAYERS - 1) * shift_per_layer) + ANIMATION_STEPS + 100  # +50 to repeat the final image to remain visible in the video
        draw_objects_shift_layers = np.ones(draw_objects_shift.shape[:-1] + (total_animation_steps,)) * np.nan  # nan value by default to tell the drawer not to draw the polygon

        # for every layer, shift the draw_objects_shift in order to have a big matrix where each step describes how to draw each polygon and what amount of shift needed to animate it through the random path
        for layer in range(LAYERS):
            shift = layer * shift_per_layer
            draw_objects_shift_layers[layer::LAYERS, :, shift:shift + ANIMATION_STEPS] = draw_objects_shift[layer::LAYERS]
            draw_objects_shift_layers[layer::LAYERS, :, shift + ANIMATION_STEPS:] = 0

        shared_worker = partial(worker, draw_objects=draw_objects,
                                draw_objects_shift_layers=draw_objects_shift_layers,
                                canvas_shape=canvas_shape,
                                video_frames_dist=video_frames_dist)
        with Pool(10) as p:
            for _ in tqdm(p.imap_unordered(shared_worker, range(total_animation_steps)), total=total_animation_steps):
                pass


render_video_frames(video_frames_dist="demos/temp/",
                    source_draw_objects_dict=os.path.join("generated_images", "draw_objects.pickle"))
