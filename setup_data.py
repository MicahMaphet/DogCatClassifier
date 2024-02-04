import os
import shutil


if not os.path.isdir('train'):
    print('Creating Directories')
    os.mkdir('train')
    os.mkdir('train/cat')
    os.mkdir('train/dog')
 
    for file_train in os.listdir('TRAIN!'):
        if 'dog' in file_train:
            shutil.move(f'TRAIN!/{file_train}', f'train/dog')
        elif 'cat' in file_train:
            shutil.move(f'TRAIN!/{file_train}', f'train/cat')
        else:
            pass
