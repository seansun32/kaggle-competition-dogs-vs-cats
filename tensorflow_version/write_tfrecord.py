import cv2
import glob
import os
import numpy as np
import tensorflow as tf
from random import shuffle



def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _make_example(mode,label,image):
    image_raw=image.tostring()
    example=tf.train.Example(features=tf.train.Features(feature={
        mode:_int64_feature(int(label)),
        'image':_bytes_feature(image_raw)}))

    return example


def _load_image(addr,img_size):
    img=cv2.imread(addr)
    img=cv2.resize(img,(img_size,img_size),interpolation=cv2.INTER_CUBIC)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img=img.astype(np.float32)
    return img


def write_tfrecords(mode,file_dir,output_tfrecord_dir,img_size):
    images=glob.glob(file_dir+'*.jpg')
    
    if mode=="train":
        shuffle(images)
        
        out_train_file=output_tfrecord_dir+'train.tfrecords'
        out_val_file=output_tfrecord_dir+'validation.tfrecords'

        train_images_split=images[0:int((1-0.2)*len(images))]
        val_images_split=images[int((1-0.2)*len(images)):]

        with tf.python_io.TFRecordWriter(out_train_file) as writer:
            i=0
            for train_file in train_images_split:
                i += 1
                if i%1000==0:
                    print(i,'train images writed')

                img=_load_image(train_file,img_size)
                if 'cat' in train_file:
                    label=1
                else:
                    label=0

                example=_make_example('label',label,img)
                writer.write(example.SerializeToString())
        
        with tf.python_io.TFRecordWriter(out_val_file) as writer:
            i=0
            for val_file in val_images_split:
                i += 1
                if i%1000==0:
                    print(i,'val images writed')

                img=_load_image(val_file,img_size)
                if 'cat' in val_file:
                    label=1
                else:
                    label=0

                example=_make_example('label',label,img)
                writer.write(example.SerializeToString())

    elif mode=="test":
        out_test_file=output_tfrecord_dir+'test.tfrecords'
        with tf.python_io.TFRecordWriter(out_test_file) as writer:
            i=0
            for test_file in images:
                filename=test_file.split('/')[-1]
                index=filename.split('.')[0]
                if i==0:
                    print(index)
                i += 1
                if i%1000==0:
                    print(i,"test images writed")
                img=_load_image(test_file,img_size)
                example=_make_example('idx',index,img)
                writer.write(example.SerializeToString())

if __name__=='__main__':
    import argparse
    
    parser=argparse.ArgumentParser(description='write tfrecord files')
    parser.add_argument("command",metavar="<command>",help="'train' or 'test'")
    parser.add_argument('--src_dir',required=True,metavar="/path/to/jpg",help="dir of jpg file ")
    parser.add_argument('--dst_dir',required=True,metavar="/path/to/out tfrecord",help="dir of outfile")
    args=parser.parse_args()

    write_tfrecords(args.command,args.src_dir,args.dst_dir,299) 
