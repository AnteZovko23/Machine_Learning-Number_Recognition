
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

from tensorflow import keras

def load_model():
    return keras.models.load_model('my_model')
# load and prepare the image
def load_image(filename):
    # load the image
    img = load_img(filename, grayscale=True, target_size=(28, 28))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 1 channel
    img = img.reshape(1, 28, 28, 1)
    # prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img

img = load_image('./sample_image.png')
model = load_model()

## Predict
digit = model.predict_classes(img)
print(digit[0])