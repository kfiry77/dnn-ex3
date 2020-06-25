from trains import Task
import tensorflow as tf
import numpy as np
from tensorflow import keras

class TensorBoardImage(keras.callbacks.TensorBoard):
    @staticmethod
    def make_image(tensor):
        from PIL import Image
        import io
        tensor = np.stack((tensor, tensor, tensor), axis=2)
        height, width, channels = tensor.shape
        image = Image.fromarray(tensor)
        output = io.BytesIO()
        image.save(output, format='PNG')
        image_string = output.getvalue()
        output.close()
        return tf.Summary.Image(height=height,
                                width=width,
                                colorspace=channels,
                                encoded_image_string=image_string)

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        super(TensorBoardImage, self).on_epoch_end(epoch, logs)
        images = self.validation_data[0]  # 0 - data; 1 - labels
        img = (255 * images[0].reshape(28, 28)).astype('uint8')

        image = self.make_image(img)
        summary = tf.Summary(value=[tf.Summary.Value(tag='image', image=image)])
        self.writer.add_summary(summary, epoch)