import math

def get_pad(input_img):
    w, h = input_img.size[:2]
    h_long = True if h > w else False

    if h_long:
        tar_size = h
    else:
        tar_size = w

    # Padding to make image as (tar_size x tar_size) pixels
    pad_left = math.ceil((tar_size - w) / 2)
    pad_right = math.floor((tar_size - w) / 2)
    pad_top = math.ceil((tar_size - h) / 2)
    pad_bottom = math.floor((tar_size - h) / 2)
    padding = (pad_left, pad_top, pad_right, pad_bottom)

    return padding