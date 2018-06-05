import os
from random import shuffle
import shutil
from sklearn.model_selection import train_test_split

def create_train_folder(dirname):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)

    os.mkdir(dirname)
    os.mkdir(dirname+'/cat')
    os.mkdir(dirname+'/dog')


if __name__=='__main__':
    import argparse
    
    parser=argparse.ArgumentParser(description='create cat&dog folders')
    parser.add_argument('--data_dir',required=True,help='path to datasets')

    args=parser.parse_args()
    
    root_prefix=args.data_dir
    train_filenames=os.listdir('%s/train/'%(root_prefix))
    test_filenames=os.listdir('%s/test/'%(root_prefix))

    my_train,my_validation=train_test_split(train_filenames,test_size=0.1,random_state=0)

    create_train_folder('%s/mytrain'%(root_prefix))
    create_train_folder('%s/myvalid'%(root_prefix))

    for filename in filter(lambda x: x.split(".")[0]=='cat',my_train):
        os.symlink('%s/train/'%(root_prefix)+filename,'%s/mytrain/cat/'%(root_prefix)+filename)

    for filename in filter(lambda x:x.split(".")[0]=='dog',my_train):
        os.symlink('%s/train/'%(root_prefix)+filename,'%s/mytrain/dog/'%(root_prefix)+filename)

    for filename in filter(lambda x:x.split(".")[0]=='cat',my_validation):
        os.symlink('%s/train/'%(root_prefix)+filename,'%s/myvalid/cat/'%(root_prefix)+filename)

    for filename in filter(lambda x:x.split('.')[0]=='dog',my_validation):
        os.symlink('%s/train/'%(root_prefix)+filename,'%s/myvalid/dog/'%(root_prefix)+filename)





