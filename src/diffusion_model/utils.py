import math 
import numpy as np 

'''
refrence: https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
'''

def exists(x):
    return x is not None 

def default(val, d):
    if exists(val):
        return val 
    return d() if callable(d) else d 

def cast_tuple(t, length = 1):
    if isinstance(t, tuple):
        return t 
    return ((t, ) * length)

def cast_list(t, length = 1):
    if isinstance(t, list):
        return t 
    return ([t] * length)

def cast_numpy_array(t, length = 1):
    if isinstance(t, np.array):
        return t 
    return np.array([t] * length)

def identity(t, *args, **kwargs):
    return t 

def cycle(dl): # yield the data in chunk of data
    while True:
        for data in dl:
            yield data 

def divisible_by(numer, denom):
    return (numer % denom) == 0

def has_int_squareroot(num): 
    '''
    check the number if it's sqaureroot is an integer

    input: 25
    output: True 

    input: 32:
    output: False 
    '''
    return math.pow((math.sqrt(num)),2) == num 

def num_to_groups(num, divisor):
    '''
    converting single number into gropus

    input: num = 10, divisor = 4
    output: [4, 4, 2]
    '''
    groups = num // divisor
    remainder = num % divisor 
    arr = [divisor] * groups 
    if remainder > 0:
        arr.append(remainder)
    return arr 

def convert_image_to_fn(img_type, image):
    '''
    convert image type
    input: img_type: "RGB", image: "gaudel.png"
    output: RGB image ("gaudel.png")
    '''
    if image.mode != img_type:
        return image.convert(img_type)
    return image 

def normalize_to_neg_one_to_one(img):
    '''
    Image normalization between -1 to 1 for all pixel values.
    Input image's pixels should be in between 0 to 1. 
    '''

    return img *2 -1 

def unnormalize_to_zero_to_one(img):
    '''
    Image normalization between 0 to 1 for all pixel values.
    Input image's pixels should be in between -1 to 1. 
    '''
    return (img + 1) * 0.5 
