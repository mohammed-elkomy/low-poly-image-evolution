import os
import cv2


#####################################################################################
################################ Detect duplicates ##################################
#####################################################################################
# To generate nice visualizations I first remove repeated images using dhashing
# https://www.pyimagesearch.com/2020/04/20/detect-and-remove-duplicate-images-from-a-dataset-for-deep-learning/

# then I remove duplicate images if they are only subsequent to each others

def dhash(image, hashSize=8):
    # convert the image to grayscale and resize the grayscale image,
    # adding a single column (width) so we can compute the horizontal
    # gradient
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hashSize + 1, hashSize))
    # compute the (relative) horizontal gradient between adjacent
    # column pixels
    diff = resized[:, 1:] > resized[:, :-1]
    # convert the difference image to a hash and return it
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])


def remove_duplicate_images(root_image_path):
    hashes = {}
    # loop over our image paths
    images = os.listdir(root_image_path)
    print(len(images))
    for imagePath in images:
        # load the input image and compute the hash
        if ".png" in imagePath:
            imagePath = os.path.join(root_image_path, imagePath)
            image = cv2.imread(imagePath)
            h = dhash(image)
            # grab all image paths with that hash, add the current image
            # path to it, and store the list back in the hashes dictionary
            p = hashes.get(h, [])
            p.append(imagePath)
            hashes[h] = p

    for key in hashes:
        print(key, hashes[key])

    # Remove subsequent duplicates 

    delete = []
    for key in hashes:
        dups = sorted(hashes[key], key=lambda item: os.path.split(item)[-1])
        for first, second in zip(dups[:-1], dups[1:]):
            first_num = int(os.path.split(first)[-1].split(".")[0])
            second_num = int(os.path.split(second)[-1].split(".")[0])
            if second_num == first_num + 1:
                delete.append(second)

    for delete_i in delete:
        os.remove(delete_i)


remove_duplicate_images(root_image_path="demos/temp/")
