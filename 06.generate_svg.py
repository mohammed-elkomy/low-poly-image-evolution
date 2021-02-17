#####################################################################################
############################## Generate svg demo ####################################
#####################################################################################
import os
import pickle

import drawSvg as draw
import numpy as np
from colormap import rgb2hex
from reportlab.graphics import renderPDF
from svglib.svglib import svg2rlg

MAX_COLOR = 255


def draw_svg_polygon(canvas, polygon, BGR_color, alpha):
    """
    Draws an irregular polygon to a canvas
    :param canvas: the svg drawing canvas
    :param polygon: the polygon points
    :param BGR_color: BGR color tuple of values 0 => 255
    :param alpha: opacity 0 -> 1 opacity
    """
    points = polygon.ravel().tolist()
    canvas.append(draw.Lines(*points,
                             close=False,
                             fill=rgb2hex(*BGR_color[::-1]), fill_opacity=alpha,
                             ))


def generate_svg(root_image_path, svg_save_path):
    with open(os.path.join(root_image_path, "draw_objects.pickle"), 'rb') as f:
        draw_objects = pickle.load(f)

        canvas_shape = (0, 0)  # automatically find the shape through maximizing
        for draw_object in draw_objects:
            points = draw_object["points"]
            canvas_shape = np.maximum(canvas_shape, points.max(axis=0))
        canvas_shape = canvas_shape.astype(np.int)

        image = draw.Drawing(*canvas_shape, displayInline=False)
        image.append(draw.Rectangle(0, 0, *canvas_shape, fill='black'))

        # shuffle(draw_objects) # don't shuffle the polygons :), they are overlaid over each other
        for idx, draw_object in enumerate(draw_objects):
            polygon = draw_object["points"]
            color = draw_object["color"]
            polygon = polygon.astype(np.int)
            color = color.astype(np.int)
            color = color.tolist()
            BGR, alpha = color[:-1], color[-1] / MAX_COLOR

            draw_svg_polygon(image, polygon, BGR, alpha)

        temp_svg = "/tmp/temp.svg"
        image.saveSvg(temp_svg)
        drawing = svg2rlg(temp_svg)
        os.remove(temp_svg)

        drawing.transform = (1, 0,
                             0, -1,
                             0, canvas_shape[1])  # reflect and translate

        renderPDF.drawToFile(drawing, svg_save_path)


generate_svg(root_image_path="demos/animate polygons", svg_save_path="demos/svg lisa.pdf")
