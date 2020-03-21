# Create Model
# ------------
import tensorflow as tf

img_h = 256
img_w = 256

def create_model(depth, start_f, num_classes, dynamic_input_shape):
    model = tf.keras.Sequential()

    # Encoder
    # -------
    for i in range( depth ):

        if i == 0:
            if dynamic_input_shape:
                input_shape = [None, None, 3]
            else:
                input_shape = [img_h, img_w, 3]
        else:
            input_shape = [None]

        model.add( tf.keras.layers.Conv2D( filters=start_f,
                                           kernel_size=(3, 3),
                                           strides=(1, 1),
                                           padding='same',
                                           input_shape=input_shape ) )
        model.add( tf.keras.layers.ReLU() )
        model.add( tf.keras.layers.MaxPool2D( pool_size=(2, 2) ) )

        start_f *= 2

    # Decoder
    # -------
    for i in range( depth ):
        model.add( tf.keras.layers.UpSampling2D( 2, interpolation='bilinear' ) )
        model.add( tf.keras.layers.Conv2D( filters=start_f // 2,
                                           kernel_size=(3, 3),
                                           strides=(1, 1),
                                           padding='same' ) )

        model.add( tf.keras.layers.ReLU() )

        start_f = start_f // 2

    # Prediction Layer
    # ----------------
    model.add( tf.keras.layers.Conv2D( filters=num_classes,
                                       kernel_size=(1, 1),
                                       strides=(1, 1),
                                       padding='same',
                                       activation='sigmoid'))

    return model