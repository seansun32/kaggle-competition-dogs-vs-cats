# kaggle-competition-dogs-vs-cats
my code for kaggle competition dogs-vs-cats

## description
two implementation for kaggle competition dogs-vs-cats

*   tensorflow_version a is pure tensorflow api implementation, with 5 conv layers
which achieves 84% validation accuracy because of non-deep model

*   keras_transfer_learning_version uses keras api and transfering
with xception model, which acheives 96% validation accuracy

### usage

#### tensorflow_version

*   ##### convert images to tfrecord file

```python
#convert train images to train_tfrecords and validation_tfrecords
python write_tfrecord.py train --stc_dir=/dir/to/train/img/ --dis_dir=/path/to/tfrecord/

#convert test images to test_tfrecords
python write_tfrecord.py train --stc_dir=/dir/to/test/img/ --dis_dir=/path/to/tfrecord/
```


*   ##### run model

```python
#train model, --step means how many steps you wanna train
python train_cnn.py train --model_dir=/path/to/save/model/ --input=/path/to/train/tfrecord/ --step=2000

#evaluate model with validation tfrecord
python train_cnn.py evaluate --model_dir=/path/to/save/model/ --input=/path/to/validation/tfrecord/

#test model on test tfrecord and a generate submission_file.csv file
python train_cnn.py test --model_dir=/path/to/save/model/ --input=/path/to/test/tfrecord/

#predict a picture whether it's a cat or dog
python train_cnn.py predict --model_dir=/path/to/save/model/ --input=/path/to/image/
```


#### keras_transfer_learning_version

*   ##### preprocess

```python
#move train images to cat/ and dog/ dirs
python preprocess.py --data_dir=/path/to/datasets
```

*   ##### run modeel

```python
#train model
python xception_keras.py train --data_dir=/path/to/datasets

#test model
python xception_keras.py test --data_dir=/path/to/datasets
```

