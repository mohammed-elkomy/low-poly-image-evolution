import glob
import os
import shutil
import numpy as np


#####################################################################################
#################### Exponentially decaying frame rate ##############################
#####################################################################################
# what I mean by exponentially is to drop more frames as I reach the end of the video, making the video faster as time progresses,
# because most of the changes happens at the beginning and the changes at the end of the video are barely noticed
# by splitting the video frames into sqr(len(frames)) chunks and drop 0 from the first chunk,1 from the second chunk and so forth

def drop_n_frames(frames_list, number_to_drop):
    """
    A very annoying way to drop inner frames and keep it consistent as possible
    :param frames_list: input frames
    :param number_to_drop: number of frames to drop
    :return: the consistent sequence of frames
    """
    if number_to_drop:
        alternating = frames_list[::2]
        remaining_to_drop = number_to_drop - len(alternating)
        if remaining_to_drop > 0:
            frames_list = drop_n_frames(alternating, remaining_to_drop)
        else:
            frames_list = frames_list[:-remaining_to_drop * 2] + alternating[-remaining_to_drop:]
    return frames_list


def drop_frames_exponentially(source_root, dist_root):
    png_images = sorted(glob.glob(os.path.join(source_root, "*.png")))

    # dropping half of the frames, in a linearly increasing profile
    keep_images = []
    num_frames = len(png_images)
    chunk_size = int(np.sqrt(num_frames))
    current_image_idx = 0
    for chunk_idx, chunk_start in enumerate(range(0, num_frames, chunk_size)):
        keep_chunk = png_images[chunk_start:chunk_start + chunk_size]
        keep_chunk = drop_n_frames(keep_chunk, chunk_idx)
        if not keep_chunk:  # no empty chunks allowed
            mid_idx = min(len(png_images) - 1, chunk_start * chunk_idx // 2)
            keep_chunk = [png_images[mid_idx]]  # the center element only

        for source_image_path in keep_chunk:
            dist_image_path = os.path.join(dist_root, "{:05d}.png".format(current_image_idx))
            current_image_idx += 1
            shutil.copy(source_image_path, dist_image_path)

    for _ in range(20):
        dist_image_path = os.path.join(dist_root, "{:05d}.png".format(current_image_idx))
        current_image_idx += 1
        shutil.copy(source_image_path, dist_image_path)


########## UNCOMMENT THIS TO drop_frames_exponentially
drop_frames_exponentially(source_root="demos/temp",  # #  source_root = "demos/animate polygons"
                          dist_root="demos/ttt")
