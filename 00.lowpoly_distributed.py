import ctypes
import multiprocessing as mp
import random
from contextlib import closing
from functools import partial
import os
import cv2
import imutils
import numpy as np
from scipy.optimize import dual_annealing
import time
import pickle

SLEEP_FACTOR = 0
MINUTE = 60

DIVISIONS_VER = 4
DIVISIONS_HOR = 4

INTERNAL_MAX_SIDE = 150


def split_with_clip(numpy_array, divisions, axis):
    """
    split and image but guarantees it's divisible by clipping extra points at the edge  
    :param numpy_array: the numpy array ( an image in this case)
    :param divisions: number of divisions
    :param axis: the axis on which the splitting is done
    :return: a list of sub-arrays
    """
    residu = original_image.shape[axis] % divisions
    residu_1 = residu // 2
    residu_2 = residu - residu_1
    if axis == 0:
        numpy_array = numpy_array[residu_1:numpy_array.shape[axis] - residu_2]
    elif axis == 1:
        numpy_array = numpy_array[:, residu_1:numpy_array.shape[axis] - residu_2]
    return np.split(numpy_array, divisions, axis=axis)


def resize_max_side(image):
    """
    resizes the maximim side of an image to a certain value
    :param image: the image to be resized
    :return: the resized image

    # INTERNAL_MAX_SIDE is the max side used, and set as constant
    """
    image_width = image.shape[1]
    image_height = image.shape[0]

    if image_width > image_height:
        image = imutils.resize(image, width=INTERNAL_MAX_SIDE).astype(np.float)
    else:
        image = imutils.resize(image, height=INTERNAL_MAX_SIDE).astype(np.float)

    return image


class LowPolyOptimizer:
    def __init__(self, reference_image, live_draw_portion, hidden_draw_portion,
                 rgba=4,  # color space used
                 max_color=255,
                 dimensions=2,  # working in 2d images
                 num_poly=250,  # number of polygons = 30, for example if POINTS_PER_POLYGON = 3, we will have 30 triangles
                 points_per_poly=4,  # 3 is a triangle
                 ):
        """
        This object is responsible for reconstructing an image / sub-image, for the example I made, I used 16 portions which in turn creates 16 instances of LowPolyOptimizer,
        that's why this class takes the live_draw_portion to draw the live reconstructed images to a sub-array (the sub-image of the whole image)
        :param reference_image: the reference image to be reconstructed
        :param live_draw_portion: a reference to the shared drawing canvas accessed by many LowPolyOptimizers
        :param hidden_draw_portion: a reference to the shared drawing canvas accessed by many LowPolyOptimizers (to be saved to the disk without overlaying the error and current step)
        :param rgba: 4, the color space, consider this as constant
        :param max_color:255, the maximum value for the color  , consider this as constant
        :param dimensions:2 working in 2d images, consider this as constant
        :param num_poly: number of polygons = 30, for example if POINTS_PER_POLYGON = 3, we will have 30 triangles
        :param points_per_poly: number of points per a polygon, 3 is a triangle
        """
        self.resized_reference_image = resize_max_side(reference_image)

        self.live_draw_portion = live_draw_portion
        self.hidden_draw_portion = hidden_draw_portion

        self.reference_shape = reference_image.shape
        self.internal_shape = self.resized_reference_image.shape

        self.max_color = max_color

        self.colors_tensor_shape = (num_poly, rgba)
        self.polygons_tensor_shape = (num_poly, points_per_poly, dimensions)

    @staticmethod
    def encode(polygons, colors):
        """
        to encode polygons and colors into SFS state vector (squashing into a big vector)
        :param polygons: numpy (NUM_OF_POLY, POINTS_PER_POLYGON, DIMENSIONS,), the polygons to draw
        :param colors: the color of every polygon (overlaid using Alpha channel)
        :return: state vector for the optimizer
        """
        squashed_polygons = polygons.reshape(-1)
        squashed_colors = colors.reshape(-1)
        return np.concatenate([squashed_polygons, squashed_colors])

    def decode(self, state_vector):
        """
        converts the state vector into polygons and colors to be drawn by opencv canvas
        :param state_vector: the input state vector to be converted
        :return: both the polygons and colors (correspond color (RGBA) for each polygon, they are overlaid over each others)
        """
        polygons_length = np.prod(self.polygons_tensor_shape)
        colors_length = np.prod(self.colors_tensor_shape)
        polygons = state_vector[:polygons_length]
        polygons = polygons.reshape(self.polygons_tensor_shape)
        colors = state_vector[-colors_length:]
        colors = colors.reshape(self.colors_tensor_shape)
        return polygons, colors

    def custom_fitness(self, state_vector):
        """
        the function that guides the search algorithm, producing and error for every reconstructed image,
        this uses the L2 distance in a normalized form, and also the sizes of polygons (I consider large polygons as bad polygons because they may ignore some details)
        :param state_vector: the input state vector proposed by the optimizer
        :return: the corresponding error, this is done throw drawing those polygons and computing the L2 distance with the reference image
        """
        polygons, colors = self.decode(state_vector)
        drawn, polygons_area_loss = self.draw_image_from_polygons(polygons, colors, canvas_shape=self.internal_shape)
        # drawn = cv2.cvtColor(drawn, cv2.COLOR_BGR2HSV) # I tried HSV color space but the results are just worse :(
        diff = self.resized_reference_image - drawn.astype(np.float32)
        pixel_error = np.mean(np.multiply(diff, diff)) / self.max_color / self.max_color  # 0 ~ 1
        return pixel_error + .3 * polygons_area_loss

    def get_generation_callback(self):
        generation = 1
        drawer = partial(self.draw_image_from_polygons, write_to_canvas=True, canvas_shape=self.reference_shape)

        last_time_callback = 0

        def generation_callback(best_point, fitness, context):
            nonlocal generation
            nonlocal last_time_callback
            polygons, colors = self.decode(best_point)
            draw_canvas = drawer(polygons, colors, error_string="{:.4f}".format(fitness))
            draw_canvas[:] = cv2.putText(draw_canvas, "Iter: {}".format(generation), (0, draw_canvas.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA, bottomLeftOrigin=False)

            if time.time() - last_time_callback > MINUTE * .25:
                if np.random.random() < SLEEP_FACTOR:
                    draw_canvas[:] = cv2.putText(draw_canvas, "ZZzz..", (0, draw_canvas.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA, bottomLeftOrigin=False)
                    time.sleep(MINUTE * .25)

                last_time_callback = time.time()
            generation += 1

        return generation_callback

    def draw_image_from_polygons(self, polygons, colors, canvas_shape, write_to_canvas=False, error_string=""):
        """
        :param polygons: numpy (NUM_OF_POLY, POINTS_PER_POLYGON, DIMENSIONS,), the polygons to draw
        :param colors: the color of every polygon (overlaid using Alpha channel)
        :param canvas_shape: image shape to draw ( changed for the demo to look large enough)
        :param write_to_canvas: to show the images
        :param error_string: the value of the fitness function formatted into string
        :return: 2d BGR image drawn, the maximum polygon area
        """

        canvas_area = np.prod(canvas_shape[:-1])

        # rescale the points for the demos
        if canvas_shape != self.internal_shape:
            polygons[..., 0] = polygons[..., 0] / self.internal_shape[0] * canvas_shape[0]
            polygons[..., 1] = polygons[..., 1] / self.internal_shape[1] * canvas_shape[1]

        # the draw canvas, drawing each polygon one by one
        image = np.zeros(canvas_shape, dtype=np.uint8)
        polygons_area = []
        for polygon, color in zip(polygons, colors):
            polygon = polygon.astype(np.int)
            color = color.astype(np.int)
            overlay = image.copy()
            color = color.tolist()
            BGR, alpha = color[:-1], color[-1] / self.max_color
            area = cv2.contourArea(polygon) / canvas_area
            polygons_area.append(area)
            cv2.drawContours(overlay, [polygon], 0, BGR, -1)
            # apply the overlay
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

        # write to the drawing canvas, this is a shared big array accessed by multiple processes, a shared reference
        if write_to_canvas:
            self.hidden_draw_portion[:] = image.copy()
            image = cv2.putText(image, error_string, (0, 23), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA, bottomLeftOrigin=False)
            self.live_draw_portion[:] = image
            return self.live_draw_portion

        return image, np.max(polygons_area)

    def optimize(self):
        """
        the optimization process, a very long time consuming task,
        it first it sets the limits for the color and the polygon points and then instantiates instance of the dual_annealing optimzier
        :return:
        """
        image_width = self.resized_reference_image.shape[1]
        image_height = self.resized_reference_image.shape[0]

        lower_polygons = np.zeros(shape=self.polygons_tensor_shape, )
        lower_colors = np.zeros(shape=self.colors_tensor_shape)
        lower_colors[:, -1] = 20
        lower = self.encode(lower_polygons, lower_colors)

        upper_polygons = np.ones(shape=self.polygons_tensor_shape, )
        upper_polygons[..., 0] = image_width  # x coordinate, the upper bound for the search algorithm(to be within the canvas)
        upper_polygons[..., 1] = image_height  # y coordinate, the upper bound for the search algorithm(to be within the canvas)
        upper_colors = np.ones(shape=self.colors_tensor_shape) * self.max_color
        upper = self.encode(upper_polygons, upper_colors)

        optimal = dual_annealing(self.custom_fitness, bounds=list(zip(lower, upper)), callback=self.get_generation_callback(), maxiter=50)
        optimal_point = optimal.x
        return self.decode(optimal_point)


def worker(args):
    """
    The worker function for each process
    :param args: the parameters,namely,
        reference_crop: the reference image we are trying to reconstruct
        hor_slice: the horizontal slice of the shared canvases (live_demo_shared_canvas,hidden_shared_canvas)
        ver_slice: the vertical slice of the shared canvases (live_demo_shared_canvas,hidden_shared_canvas)
        shared_canvas_shape the shared canvas shape used to restore the binary array data into a numpy ndarray
    :return: a dictionary of the polygon shapes with their corresponding RGBA
    """
    # https://stackoverflow.com/questions/32172054/how-can-i-retrieve-the-current-seed-of-numpys-random-number-generator
    np.random.seed(None)  # numpy must be seeded in every process, because child processes inherit the same state

    reference_crop, hor_slice, ver_slice, shared_canvas_shape = args

    shared_canvas_reference = bytes_to_image(live_demo_shared_canvas, shared_canvas_shape)
    hidden_canvas_reference = bytes_to_image(hidden_shared_canvas, shared_canvas_shape)

    live_demo_canvas_portion = shared_canvas_reference[ver_slice, hor_slice]
    hidden_canvas_portion = hidden_canvas_reference[ver_slice, hor_slice]
    optimizer = LowPolyOptimizer(reference_crop, live_demo_canvas_portion, hidden_canvas_portion)
    polygons, colors = optimizer.optimize() # A VERY LOOONG CALL
    # rescale the image
    polygons[..., 0] = polygons[..., 0] / optimizer.internal_shape[0] * optimizer.reference_shape[0]
    polygons[..., 1] = polygons[..., 1] / optimizer.internal_shape[1] * optimizer.reference_shape[1]
    shifted_polygons = polygons + (hor_slice.start, ver_slice.start)  # position each sub-image in the whole drawing canvas

    poly_objects = []
    for polygon, color in zip(shifted_polygons, colors):
        poly_objects.append({"points": polygon, "color": color})
    return poly_objects


def init(live_demo_shared_canvas_init, hidden_shared_canvas_init):
    """
    init function for the pool of processes having a shared canvas to draw the reconstructed portions, where each process works on a portion of this canvas
    :param live_demo_shared_canvas_init: the canvas used for the live demo, shown on the desktop by the main thread
    :param hidden_shared_canvas_init: the hidden canvas to be saved to the disk (the one saved to the disk doesn't have the fitness and iteration number overlaid on the image as in the live demo)
    :return:
    """
    global live_demo_shared_canvas
    global hidden_shared_canvas
    live_demo_shared_canvas = live_demo_shared_canvas_init  # must be inherited, not passed as an argument
    hidden_shared_canvas = hidden_shared_canvas_init  # must be inherited, not passed as an argument


def bytes_to_image(mp_arr, shape):
    """
    the shared array between the multiple processes has to be in the binary form
    :param mp_arr: this is the binary form of the numpy array, shared by multiprocessing package
    :param shape: the target shape for the array
    :return: mp_arr converted into a numpy array to be accessed by opencv
    """
    return np.frombuffer(mp_arr.get_obj(), dtype=np.uint8).reshape(shape)


if __name__ == "__main__":
    # load the image
    original_image = cv2.imread("imgs/liza.jpg")

    # clip indivisible pixel positions
    shared_canvas_shape = (original_image.shape[0] - original_image.shape[0] % DIVISIONS_VER,
                           original_image.shape[1] - original_image.shape[1] % DIVISIONS_HOR,
                           3)

    hor_splits = split_with_clip(original_image, DIVISIONS_VER, axis=0)
    params = []

    for column, hor_split in enumerate(hor_splits):
        ver_splits = split_with_clip(hor_split, DIVISIONS_HOR, axis=1)
        for row, ver_split in enumerate(ver_splits):
            height, width, _ = ver_split.shape
            hor_slice = slice(row * width, (row + 1) * width)
            ver_slice = slice(column * height, (column + 1) * height)
            params.append((ver_split, hor_slice, ver_slice, shared_canvas_shape))

    live_demo_shared_canvas = mp.Array(ctypes.c_uint8, int(np.prod(shared_canvas_shape)))
    hidden_shared_canvas = mp.Array(ctypes.c_uint8, int(np.prod(shared_canvas_shape)))
    live_canvas = bytes_to_image(live_demo_shared_canvas, shared_canvas_shape)
    hidden_canvas = bytes_to_image(hidden_shared_canvas, shared_canvas_shape)

    # write to arr from different processes
    # async mp + sharing: https://stackoverflow.com/questions/7894791/use-numpy-array-in-shared-memory-for-multiprocessing/
    with closing(mp.Pool(processes=DIVISIONS_VER * DIVISIONS_HOR, initializer=init, initargs=(live_demo_shared_canvas, hidden_shared_canvas))) as p:
        # many processes access different slices of the same array
        results = p.map_async(worker, params)

    time.sleep(5)
    idx = 0
    last_time = 0
    print("press Q for 3 seconds to kill the process")
    quit_count = 0
    while not results.ready():

        if time.time() - last_time > MINUTE * .1:
            # write image every 200 seconds
            idx += 1
            cv2.imwrite(os.path.join("generated_images", "{:05d}.png".format(idx)), hidden_canvas)
            last_time = time.time()

        cv2.imshow("Reconstructed", live_canvas)
        key = chr(cv2.waitKey(1000) & 0xFF)
        if key == "q":
            quit_count += 1
            print("remaining {} seconds to exit".format(3 - quit_count))
        else:
            quit_count = 0

        if quit_count == 3:
            print("User quited on purpose")
            exit()

    distributed_draw_objects = results.get()  # returned as a result list from N workers

    draw_objects = []
    for draw_objects_i in distributed_draw_objects:
        draw_objects.extend(draw_objects_i)

    with open(os.path.join("generated_images", "draw_objects.pickle"), 'wb') as f:
        pickle.dump(draw_objects, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("DONE")
    cv2.waitKey()
