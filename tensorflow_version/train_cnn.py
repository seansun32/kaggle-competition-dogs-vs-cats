import tensorflow as tf
import cv2
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

LEARNING_RATE=0.001
NUM_STEPS=2000
BSTCH_SIZE=128

NUM_INPUT=89401 #299*299
NUM_CLASSES=2
DROPOUT=0.25


#--------------input function---------
def parse_image(file_name):
    img=cv2.imread(file_name)
    img=cv2.resize(img,(299,299),interpolation=cv2.INTER_CUBIC)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img=img.astype(np.float32)
    return img

def train_parse_record(raw_record):
    features=tf.parse_single_example(
            raw_record,
            features={
                'label':tf.FixedLenFeature([],tf.int64),
                'image':tf.FixedLenFeature([],tf.string)})

    images=tf.decode_raw(features['image'],tf.float32)
    labels=tf.cast(features['label'],tf.int32)

    return {'image':images},labels

def test_parse_record(raw_record):
    features=tf.parse_single_example(
            raw_record,
            features={
                'idx':tf.FixedLenFeature([],tf.int64),
                'image':tf.FixedLenFeature([],tf.string)})
    images=tf.decode_raw(features['image'],tf.float32)
    index=tf.cast(features['idx'],tf.int32)

    return {'image':images,'idx':index}


def input_fn(file_name,batch_size,num_epochs=5,num_parallel_calls=1):
    dataset=tf.data.TFRecordDataset(file_name)

    dataset=dataset.map(lambda value:train_parse_record(value),num_parallel_calls=num_parallel_calls)

    dataset=dataset.shuffle(buffer_size=10000)
    dataset=dataset.batch(batch_size)
    dataset=dataset.repeat(num_epochs)
    iterator=dataset.make_one_shot_iterator()

    features,labels=iterator.get_next()
    return features,labels

def train_input_fn(file_path):
    return input_fn(file_path,100,None,10)

def validation_input_fn(file_path):
    return input_fn(file_path,50,1,1)

def test_input_fn(file_path):
    dataset=tf.data.TFRecordDataset(file_path)
    dataset=dataset.map(lambda value:test_parse_record(value),
                        num_parallel_calls=1)

    dataset=dataset.batch(1)
    iterator=dataset.make_one_shot_iterator()
    features=iterator.get_next()
    
    return features

def predict_input_fn(file_name):

    img=parse_image(file_name)
    x={'image':np.array([img]),'idx':[0]}
    dataset=tf.data.Dataset.from_tensor_slices(x)
    dataset=dataset.batch(1)
    iterator=dataset.make_one_shot_iterator()
    one_element=iterator.get_next()
    
    return one_element



def model_fn(features,labels,mode):

    x=tf.reshape(features['image'],shape=[-1,299,299,3])
    
    #-------model structure-------------
    conv1=tf.layers.conv2d(x,32,5,activation=tf.nn.relu)
    conv1=tf.layers.max_pooling2d(conv1,2,2)

    conv2=tf.layers.conv2d(conv1,64,3,activation=tf.nn.relu)
    conv2=tf.layers.max_pooling2d(conv2,2,2)

    conv3=tf.layers.conv2d(conv2,128,5,activation=tf.nn.relu)
    conv3=tf.layers.max_pooling2d(conv3,2,2)

    conv4=tf.layers.conv2d(conv3,64,5,activation=tf.nn.relu)
    conv4=tf.layers.max_pooling2d(conv4,2,2)

    conv5=tf.layers.conv2d(conv4,32,3,activation=tf.nn.relu)
    conv5=tf.layers.max_pooling2d(conv5,2,2)

    fc1=tf.contrib.layers.flatten(conv5)
    fc1=tf.layers.dense(fc1,1024,activation=tf.nn.relu)
    fc1=tf.layers.dropout(fc1,rate=DROPOUT,training=mode==tf.estimator.ModeKeys.TRAIN)
    
    logits=tf.layers.dense(fc1,NUM_CLASSES)

    #---------end model structure------------
    

    predictions={
            "features":features['image'],
            "classes":tf.argmax(input=logits,axis=1),
            "probabilities":tf.nn.softmax(logits,name="softmax_tensor")}


    #prediction mode
    if mode==tf.estimator.ModeKeys.PREDICT:

        predictions={
            "index":features['idx'],
            "features":features['image'],
            "classes":tf.argmax(input=logits,axis=1),
            "probabilities":tf.nn.softmax(logits,name="softmax_tensor")}

        return tf.estimator.EstimatorSpec(mode=mode,predictions=predictions)

    
    #train mode
    onehot_labels=tf.one_hot(indices=tf.cast(labels,tf.int32),depth=2)
    loss=tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels,logits=logits)
    
    if mode==tf.estimator.ModeKeys.TRAIN:

        optimizer=tf.train.AdamOptimizer(learning_rate=0.001)
        train_op=optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode=mode,loss=loss,train_op=train_op)


    #evaluate mode
    eval_metric_ops={
            "accuracy":tf.metrics.accuracy(
                labels=labels,
                predictions=predictions["classes"])}

    return tf.estimator.EstimatorSpec(
            mode=mode,loss=loss,eval_metric_ops=eval_metric_ops)


if __name__=='__main__':
    import argparse

    parser=argparse.ArgumentParser(description='train/test/predict model')
    parser.add_argument("command",metavar="<command>",help='train/test/evaluate/predict')
    parser.add_argument('--model_dir',required=True,metavar='/path/to/model',help='path to model')
    parser.add_argument('--step',type=int,help='train steps')
    parser.add_argument('--input',required=True,help='path to tfrecords')
    

    args=parser.parse_args()
    
    model=tf.estimator.Estimator(model_fn=model_fn,model_dir=args.model_dir)
    tensors_to_log={"probabilities":"softmax_tensor"}
    logging_hook=tf.train.LoggingTensorHook(tensors=tensors_to_log,every_n_iter=50)
    
    if args.command=='train':
        #pass
        if args.step is None:
            parser.error("train mode requires --step")
        model.train(input_fn=lambda:train_input_fn(args.input),steps=args.step,hooks=[logging_hook])
        #e=model.evaluate(input_fn=lambda:validation_input_fn(args.input))
        #print(e)
    elif args.command=='evaluate':
        e=model.evaluate(input_fn=lambda:validation_input_fn(args.input))
        print(e)

    elif args.command=='test':
        #pass
        results=model.predict(input_fn=lambda:test_input_fn(args.input))
        with open('submission_file.csv','w') as f:
            f.write('id,label\n')
        with open('submission_file.csv','a') as f:
            for result in results:
                index=result['index']
                label=result['classes']
                prob=result['probabilities'][0]

                f.write('{},{}\n'.format(index,prob))

    elif args.command=='predict':
        #pass
        results=model.predict(input_fn=lambda:predict_input_fn(args.input))
        for result in results:
            label=result['classes']
            prob=result['probabilities']
            img=result['features']
            print("prob:",prob) 
            if label==0:
                print('The picture is a dog')
            else:
                print('The picture is a cat')



