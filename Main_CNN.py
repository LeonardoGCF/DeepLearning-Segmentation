from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

from CNN import create_model
from splitValidationSet import *
from PIL import Image

import scipy as sp
import os
import tensorflow as tf
import numpy as np
import pandas as pd
# Set the seed for random operations.
# This let our experiments to be reproducible.
SEED = 1234
tf.random.set_seed(SEED)

# Get current working directory
cwd = os.getcwd()

# Set GPU memory growth
# Allows to only as much GPU memory as needed
#gpus = tf.config.experimental.list_physical_devices('GPU')
#if gpus:
#  try:
#    # Currently, memory growth needs to be the same across GPUs
#    for gpu in gpus:
#      tf.config.experimental.set_memory_growth(gpu, True)
#    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#  except RuntimeError as e:
#    # Memory growth must be set before GPUs have been initialized
#    print(e)
#
# ImageDataGenerator
# ------------------

from tensorflow.keras.preprocessing.image import ImageDataGenerator

apply_data_augmentation = False

# Create training ImageDataGenerator object
# We need two different generators for images and corresponding masks
if apply_data_augmentation:
    train_img_data_gen = ImageDataGenerator(rotation_range=10,
                                            width_shift_range=10,
                                            height_shift_range=10,
                                            zoom_range=0.3,
                                            horizontal_flip=True,
                                            vertical_flip=True,
                                            fill_mode='constant',
                                            cval=0,
                                            rescale=1./255)
    train_mask_data_gen = ImageDataGenerator(rotation_range=10,
                                             width_shift_range=10,
                                             height_shift_range=10,
                                             zoom_range=0.3,
                                             horizontal_flip=True,
                                             vertical_flip=True,
                                             fill_mode='constant',
                                             cval=0,
                                             rescale=1./255)
else:
    train_img_data_gen = ImageDataGenerator(rescale=1./255)
    train_mask_data_gen = ImageDataGenerator(rescale=1./255)

# Create validation and test ImageDataGenerator objects
valid_img_data_gen = ImageDataGenerator(rescale=1./255)
valid_mask_data_gen = ImageDataGenerator(rescale=1./255)
test_img_data_gen = ImageDataGenerator(rescale=1./255)
test_mask_data_gen = ImageDataGenerator(rescale=1./255)

## =================================================================
##   step 2 : Create validation set（ only  use once to produce the validation set)
## =================================================================
#filePath, child_list =eachFile("Segmentation_Dataset/training/")
#for i in child_list:
#    path = 'Segmentation_Dataset/split-train/' + i +'/img'
#    mkdir( path )
#    path = 'Segmentation_Dataset/split-validation/' + i +'/img'
#    mkdir( path )
#
#train_pic_dir = []
#validation_pic_dir = []
#
##write the pic
#pic_dir, pic_name = eachFile1("Segmentation_Dataset/training/images/img" )
#random.shuffle( pic_dir )
#train_list = pic_dir[0:int( 0.8 * len( pic_dir ) )]
#validation_list = pic_dir[int( 0.8 * len( pic_dir ) ):]
#for j in train_list:
#    fromImage = Image.open( j )
#    j=j.replace('training', 'split-train')
#    #print('pic'+j+'stored in')
#    fromImage.save( j )
#    #print(j+'successful')
#for k in validation_list:
#    fromImage = Image.open( k )
#    k=k.replace('training', 'split-validation')
#    #print('pic'+k+'stored in')
#    fromImage.save( k )
#     # print(k+'successful')
#
##STEP 3
## write the mask
#mask_dir, mask_name = eachFile1("Segmentation_Dataset/training/masks/img" )
#img_dir,img_name = eachFile1("Segmentation_Dataset/split-train/images/img")
#val_dir,val_name = eachFile1("Segmentation_Dataset/split-validation/images/img")
#
#for i in mask_name:
#    if i in img_name:
#        j=os.path.join('Segmentation_Dataset/training/masks/img', i)
#        fromImage = Image.open(j)
#        j=j.replace('training', 'split-train')
#        fromImage.save(j)
#
#for i in mask_name:
#    if i in val_name:
#        k=os.path.join('Segmentation_Dataset/training/masks/img', i)
#        fromImage = Image.open(k)
#        k=k.replace('training', 'split-validation')
#        fromImage.save(k)
#
# Create generators to read images from dataset directory
# -------------------------------------------------------
dataset_dir = os.path.join(cwd, 'Segmentation_Dataset')

# Batch size
bs = 4

# img shape
img_h = 256
img_w = 256

# Training
# Two different generators for images and masks
# ATTENTION: here the seed is important!! We have to give the same SEED to both the generator
# to apply the same transformations/shuffling to images and corresponding masks
training_dir = os.path.join(dataset_dir, 'split-train')
train_img_gen = train_img_data_gen.flow_from_directory(os.path.join(training_dir, 'images'),
                                                       target_size=(img_h, img_w),
                                                       batch_size=bs,
                                                       class_mode=None, # Because we have no class subfolders in this case
                                                       shuffle=True,
                                                       interpolation='bilinear',
                                                       seed=SEED)
train_mask_gen = train_mask_data_gen.flow_from_directory(os.path.join(training_dir, 'masks'),
                                                         target_size=(img_h, img_w),
                                                         color_mode='grayscale',
                                                         batch_size=bs,
                                                         class_mode=None, # Because we have no class subfolders in this case
                                                         shuffle=True,
                                                         interpolation='bilinear',
                                                         seed=SEED)
train_gen = zip(train_img_gen, train_mask_gen)

# Validation
validation_dir = os.path.join(dataset_dir, 'split-validation')
valid_img_gen = valid_img_data_gen.flow_from_directory(os.path.join(validation_dir, 'images'),
                                                       target_size=(img_h, img_w),
                                                       batch_size=bs,
                                                       class_mode=None, # Because we have no class subfolders in this case
                                                       shuffle=False,
                                                       interpolation='bilinear',
                                                       seed=SEED)
valid_mask_gen = valid_mask_data_gen.flow_from_directory(os.path.join(validation_dir, 'masks'),
                                                         target_size=(img_h, img_w),
                                                         batch_size=bs,
                                                         color_mode='grayscale',
                                                         class_mode=None, # Because we have no class subfolders in this case
                                                         shuffle=False,
                                                         interpolation='bilinear',
                                                         seed=SEED)
valid_gen = zip(valid_img_gen, valid_mask_gen)

## Test
test_dir = os.path.join(dataset_dir, 'test')
test_img_gen = test_img_data_gen.flow_from_directory(os.path.join(test_dir, 'images'),
                                                     target_size=(img_h, img_w),
                                                     batch_size=bs,
                                                     class_mode=None, # Because we have no class subfolders in this case
                                                     shuffle=False,
                                                     interpolation='bilinear',
                                                     seed=SEED)
#test_mask_gen = test_mask_data_gen.flow_from_directory(os.path.join(test_dir, 'masks'),
#                                                       target_size=(img_h, img_w),
#                                                       batch_size=bs,
#                                                       color_mode='grayscale',
#                                                       class_mode=None, # Because we have no class subfolders in this case
#                                                       shuffle=False,
#                                                       interpolation='bilinear',
#                                                       seed=SEED)
#test_gen = zip(test_img_gen, test_mask_gen)

#-------------------------------------------------

# Create Dataset objects
# ----------------------

# Training
# --------
train_dataset = tf.data.Dataset.from_generator(lambda: train_gen,
                                               output_types=(tf.float32, tf.float32),
                                               output_shapes=([None, img_h, img_w, 3], [None, img_h, img_w, 1]))

def prepare_target(x_, y_):
    y_ = tf.cast(y_, tf.int32)
    return x_, y_

#def prepare_target(x_, y_):
#    y_ = tf.cast(tf.expand_dims(y_[..., 0], -1), tf.int32)
#    return x_, tf.where(y_ > 0, y_ - 1, y_ + 1)

train_dataset = train_dataset.map(prepare_target)

# Repeat
train_dataset = train_dataset.repeat()

# Validation
# ----------
valid_dataset = tf.data.Dataset.from_generator(lambda: valid_gen,
                                               output_types=(tf.float32, tf.float32),
                                               output_shapes=([None, img_h, img_w, 3], [None, img_h, img_w, 1]))
valid_dataset = valid_dataset.map(prepare_target)

# Repeat
valid_dataset = valid_dataset.repeat()

# Test
# ----
#test_dataset = tf.data.Dataset.from_generator(lambda: test_gen,
#                                              output_types=(tf.float32, tf.float32),
#                                              output_shapes=([None, img_h, img_w, 3], [None, img_h, img_w, 1]))
#test_dataset = test_dataset.map(prepare_target)
#
## Repeat
#test_dataset = valid_dataset.repeat()
#===================Udet++++++++++++++++++++++++++++++++++++++++++
#from Udet import *
#model = unet()


#==================================================================
model = create_model(depth=4,
                     start_f=4,
                     num_classes=2,
                     dynamic_input_shape=False)

#=====================================================================

# Optimization params
# -------------------

# Loss
# Sparse Categorical Crossentropy to use integers (mask) instead of one-hot encoded labels
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
# learning rate
lr = 1e-3
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
# -------------------

# Validation metrics
# ------------------

metrics = ['accuracy']
# ------------------

# Compile Model
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

#=====================================================================

import os
from datetime import datetime

# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession

# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

cwd = os.getcwd()

exps_dir = os.path.join( cwd, 'segmentation_experiments' )
if not os.path.exists( exps_dir ):
    os.makedirs( exps_dir )

now = datetime.now().strftime( '%b%d_%H-%M-%S' )

model_name = 'CNN'

exp_dir = os.path.join( exps_dir, model_name + '_' + str( now ) )
if not os.path.exists( exp_dir ):
    os.makedirs( exp_dir )

callbacks = []

# Model checkpoint
# ----------------
ckpt_dir = os.path.join( exp_dir, 'ckpts' )
if not os.path.exists( ckpt_dir ):
    os.makedirs( ckpt_dir )

ckpt_callback = tf.keras.callbacks.ModelCheckpoint( filepath=os.path.join( ckpt_dir, 'cp_{epoch:02d}.ckpt' ),
                                                    save_weights_only=True )  # False to save the model directly
callbacks.append( ckpt_callback )

# Visualize Learning on Tensorboard
# ---------------------------------
tb_dir = os.path.join( exp_dir, 'tb_logs' )
if not os.path.exists( tb_dir ):
    os.makedirs( tb_dir )

# By default shows losses and metrics for both training and validation
tb_callback = tf.keras.callbacks.TensorBoard( log_dir=tb_dir,
                                              profile_batch=0,
                                              histogram_freq=0 )  # if 1 shows weights histograms
callbacks.append( tb_callback )

# Early Stopping
# --------------
early_stop = False
if early_stop:
    es_callback = tf.keras.callback.EarlyStopping( monitor='val_loss', patience=10 )
    callbacks.append(es_callback)


#=================================
model.fit( x=train_dataset,
           epochs=40,  #### set repeat in training dataset
           steps_per_epoch=len(train_img_gen),
           validation_data=valid_dataset,
           validation_steps=len(valid_img_gen),
           callbacks=callbacks
          )

# How to visualize Tensorboard

# 1. tensorboard --logdir EXPERIMENTS_DIR --port PORT     <- from terminal
# 2. localhost:PORT   <- in your browser

#========================================================================
from datetime import datetime

def create_csv(results, results_dir='./'):

    csv_fname = 'results_'
    csv_fname += datetime.now().strftime('%b%d_%H-%M-%S') + '.csv'

    with open(csv_fname, 'w') as f:

      f.write('ImageId,EncodedPixels,Width,Height\n')

      for key, value in results.items():
          f.write(key + ',' + str(value) + ',' + '256' + ',' + '256' + '\n')

def rle_encode(img):
      # Flatten column-wise
      pixels = img.T.flatten()
      pixels = np.concatenate([[0], pixels, [0]])
      runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
      runs[1::2] -= runs[::2]
      return ' '.join(str(x) for x in runs)


#from torchvision import transforms
#unloader = transforms.ToPILImage()
#def tensor_to_PIL(tensor):
#    image = tensor.cpu().clone()
#    image = image.squeeze(0)
#    image = unloader(image)
#    return image



test_img_dir = os.path.join( test_dir, 'images', 'img' )
#test_mask_dir = os.path.join( test_dir, 'masks', 'img' )

img_filenames = next( os.walk( test_img_dir ) )[2]

#fig, ax = plt.subplots( 1, 3, figsize=(8, 8) )
#fig.show()



final_result = {}


for img_filename in img_filenames:


    img = Image.open( os.path.join( test_img_dir, img_filename ) )
    img_arr = np.expand_dims( np.array( img ), 0 )
    out_softmax = model.predict( x=img_arr / 255. )

    # Get predicted class as the index corresponding to the maximum value in the vector probability
    predicted_class = tf.argmax( out_softmax, -1 )
    predicted_class = predicted_class[0]
   # transforms_predicted_class = tensor_to_PIL(predicted_class)
    result={img_filename[:-4]: rle_encode(predicted_class.numpy())}
    final_result.update(result)

create_csv(final_result)




