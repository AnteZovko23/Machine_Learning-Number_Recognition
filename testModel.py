from tensorflow import keras

from tensorflow.python.keras.preprocessing.image  import load_img
from tensorflow.python.keras.preprocessing.image  import img_to_array

from draw import start

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps, ImageChops
from io import BytesIO
## Found here: https://www.toptal.com/data-science/machine-learning-number-recognition
# Image preprocessing to make it similar to training data #############################


def replace_transparent_background(image):
    image_arr = np.array(image)

    if len(image_arr.shape) == 2:
        return image

    alpha1 = 0
    r2, g2, b2, alpha2 = 255, 255, 255, 255

    red, green, blue, alpha = image_arr[:, :, 0], image_arr[:, :, 1], image_arr[:, :, 2], image_arr[:, :, 3]
    mask = (alpha == alpha1)
    image_arr[:, :, :4][mask] = [r2, g2, b2, alpha2]

    return Image.fromarray(image_arr)



def invert_colors(image):
    return ImageOps.invert(image)

def pad_image(image):
    return ImageOps.expand(image, border=30, fill='#fff')

def trim_borders(image):
    bg = Image.new(image.mode, image.size, image.getpixel((0,0)))
    diff = ImageChops.difference(image, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return image.crop(bbox)
    
    return image


def process_image(image):

    img = replace_transparent_background(image)
    img = invert_colors(img)
    img = pad_image(img)
    img = trim_borders(img)

    return img

##############################################################




def load_model():
    return keras.models.load_model('my_model')
# load and prepare the image
def load_image(filename):
    # load the image
    img = load_img(filename, color_mode='grayscale', target_size=(28, 28))



    img = process_image(img)
    # convert to array
    img = img_to_array(img)
    img = img.reshape(1, 28, 28, 1)

    # reshape into a single sample with 1 channel
    # prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img


def show_image(img, label, guess):
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.title("Excpected: " + label)
    plt.xlabel("Guess: " + guess)
    plt.colorbar()
    plt.grid(False)
    plt.show()

while(True):
    start()
    imgPath = "./user_number.png"

    img = load_image(imgPath)

    model = load_model()

    # New syntax
    guess = np.argmax(model.predict(img), axis=-1)
    img = img.reshape(28,28, 1)
    show_image(img, imgPath[2:3], str(guess[0]))



# # Predict
# # Deprecated
# digit = model.predict_classes(img)



