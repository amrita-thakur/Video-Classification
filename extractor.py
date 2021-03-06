from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model
import numpy as np

class Extractor():
    def __init__(self, weights=None):
        """Either load pretrained from imagenet, or load our saved
        weights from our own training."""

        self.weights = weights  # so we can check elsewhere which model

        if weights is None:
            # Get model with pretrained weights.
            base_model = InceptionV3(
                weights='imagenet', #One of None (random initialization), imagenet (pre-training on ImageNet), or the path to the weights file to be loaded. Default to imagenet.
                include_top=True #Boolean, whether to include the fully-connected layer at the top, as the last layer of the network. Default to True.
            )

            # We'll extract features at the final pool layer.
            self.model = Model(
                inputs=base_model.input,
                outputs=base_model.get_layer('avg_pool').output
            )

        else:
            # Load the model first.
            self.model = load_model(weights)

            # Then remove the top so we get features not predictions.
            # From: https://github.com/fchollet/keras/issues/2371
            self.model.layers.pop()
            self.model.layers.pop()  # two pops to get to pool layer
            self.model.outputs = [self.model.layers[-1].output]
            self.model.output_layers = [self.model.layers[-1]]
            self.model.layers[-1].outbound_nodes = []

    def extract(self, image_path):
        img = image.load_img(image_path, target_size=(299, 299)) #This loads an image and resizes the image to (299, 299)
        x = image.img_to_array(img)    #The img_to_array() function adds channels: x.shape = (299, 299, 3) for RGB and (299, 299, 1) for gray image
        x = np.expand_dims(x, axis=0)  #expand the shape of an array. expand_dims() is used to add the number of images: x.shape = (1, 299, 299, 3)
        x = preprocess_input(x)  #f you add x to an array images, at the end of the loop, you need to add images = np.vstack(images) so that you get (n, 299, 299, 3) as the dim of images where n is the number of images processed
        # Keras works with batches of images. So, the first dimension is used for the number of samples (or images) you have.
        # When you load a single image, you get the shape of one image, which is (size1,size2,channels).
        # In order to create a batch of images, you need an additional dimension: (samples, size1,size2,channels)
        # The preprocess_input function is meant to adequate your image to the format the model requires.


        # Get the prediction.
        features = self.model.predict(x)

        if self.weights is None:
            # For imagenet/default network:
            features = features[0]
        else:
            # For loaded network:
            features = features[0]

        return features
