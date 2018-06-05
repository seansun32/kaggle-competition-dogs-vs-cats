from keras import applications
from keras import optimizers
from keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
from tqdm import *
import os
import cv2

import tensorflow as tf
from   keras           import backend as K
from   keras.models    import Model
from   keras.layers    import Dense, Input, BatchNormalization, Activation, merge, Dropout
from   keras.layers    import Conv2D, SeparableConv2D, MaxPooling2D, GlobalAveragePooling2D
from   keras.callbacks import ModelCheckpoint
from   keras.preprocessing       import image
from   keras.preprocessing.image import ImageDataGenerator
from   keras.engine.topology     import get_source_inputs
from   keras.utils.data_utils    import get_file
from   keras.applications.imagenet_utils import decode_predictions, _obtain_input_shape


IMAGE_WIDTH=224
IMAGE_HEIGHT=224
IMAGE_SIZE=(IMAGE_WIDTH,IMAGE_HEIGHT)

TOP_NUM=2
    
def get_image(index,data_dir):
    img=cv2.imread(data_dir+'/test/%d.jpg'%(index))
    img=cv2.resize(img,IMAGE_SIZE)
    img.astype(np.float32)
    img=img/255.0

    return img


if __name__=='__main__':
    
    import argparse
    
    parser=argparse.ArgumentParser(description='xception transfer learning')
    parser.add_argument('command',metavar='<command>',help="'train' or 'test'")
    parser.add_argument('--data_dir',required=True,help='path to datasets')
    
   
    args=parser.parse_args()


    train_datagen=ImageDataGenerator(
        rescale=1.0/255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    
    train_flow=train_datagen.flow_from_directory(
        '%s/mytrain'%(args.data_dir),
        target_size=IMAGE_SIZE,
        batch_size=16,
        class_mode='binary'
    )
    
    validation_datagen=ImageDataGenerator(rescale=1.0/255)
    validation_flow=validation_datagen.flow_from_directory(
            '%s/myvalid'%(args.data_dir),
            target_size=IMAGE_SIZE,
            batch_size=16,
            class_mode='binary'
    )

    base_model=applications.xception.Xception(weights='imagenet',include_top=False)
    x=GlobalAveragePooling2D(name='avg_pool')(base_model.output)
    x=Dropout(0.5)(x)
    x=Dense(1,activation='sigmoid',name='output')(x)
    
    model=Model(input=base_model.input,output=x)
    
    for layer in model.layers[:TOP_NUM]:
        layer.trainable=False

    for layer in model.layers[TOP_NUM:]:
        layer.trainable=True
    
    if args.command=='train':
        model.compile(loss='binary_crossentropy',optimizer='adadelta',metrics=['accuracy'])
        best_model=ModelCheckpoint('xception_best.h5',monitor='val_loss',verbose=0,save_best_only=True)

        model.fit_generator(
                train_flow,
                samples_per_epoch=2048,
                nb_epoch=25,
                validation_data=validation_flow,
                nb_val_samples=1024,
                callbacks=[best_model])

        with open('xception.json','w') as f:
            f.write(model.to_json())
    
    elif args.command=='test':
        model.load_weights('xception_best.h5')
        
        test_num=len(os.listdir(args.data_dir+'/test'))
        image_matrix = np.zeros((test_num, IMAGE_WIDTH, IMAGE_HEIGHT, 3), dtype=np.float32)
        
        for i in tqdm(range(test_num)):
            image_matrix[i]=get_image(i+1,args.data_dir)
        
        predictions=model.predict(image_matrix,verbose=1)
        s='id,label\n'
        for i,p in enumerate(predictions):
            s+='%d,%f\n'%(i+1,p)

        with open('submission.csv','w') as f:
            f.write(s)




