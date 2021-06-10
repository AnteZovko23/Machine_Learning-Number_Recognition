from tensorflow import keras

from tensorflow.python.keras.preprocessing.image  import load_img
from tensorflow.python.keras.preprocessing.image  import img_to_array

from draw import start

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps, ImageChops

import mplcursors

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



## Plots all the possibilities as a bar graph
def plot_possibilities(guess_array, ind_length = 10):

    plt.figure(100)

    ## X axis
    ind = np.arange(ind_length)
    width = 0.35

    ## Guess array rounded to 10 decimal places

    ## Reformat array to be one dimensional
    guess_array = np.array(guess_array[0])
    print(guess_array)
    # format for title
    guess_array_title = {ind[i]:f'{(guess_array[i] * 100):.5f} %' for i in range(len(guess_array))}
    

    ## Make window bigger
    manager = plt.get_current_fig_manager()
    manager.window.maximize()


    # Graph
    plt.ylim(0, 1)
    plt.title(guess_array_title)
    plt.bar(ind, guess_array, width, label="a")



    # With HoverMode.Transient, the annotation is removed as soon as the mouse
    # leaves the artist.  Alternatively, one can use HoverMode.Persistent (or True)
    # which keeps the annotation until another artist gets selected.
    cursor = mplcursors.cursor(hover=mplcursors.HoverMode.Transient)
    @cursor.connect("add")
    def on_add(sel):
        x, y, width, height = sel.artist[sel.target.index].get_bbox().bounds
        sel.annotation.set(text=f"{sel.target.index}:{round((height*100), 10)}%",
                       position=(0, 20), anncoords="offset points")
        sel.annotation.xy = (x + width / 2, y + height)

    plt.show()


# Shows the guess
def show_image(img, guess):
    plt.figure(200)
    plt.imshow(img, cmap='gray')
    plt.title("Guess: " + guess)
    plt.colorbar()
    plt.grid(False)
    

    plt.show()


model = load_model()

while(True):
    start()
    imgPath = "./user_number.png"

    img = load_image(imgPath)

    prediction = model.predict(img)

    # New syntax
    guess = np.argmax(prediction, axis=-1)
    img = img.reshape(28,28, 1)

    # Graphs!
    show_image(img, str(guess[0]))
    plot_possibilities(prediction)




# # Predict
# # Deprecated
# digit = model.predict_classes(img)



