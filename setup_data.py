import os
import shutil


if not os.path.isdir('train'):
    print('Creating Directories')
    os.mkdir('train')
    os.mkdir('test')
    os.mkdir('train/cat')
    os.mkdir('train/dog')
    os.mkdir('test/cat')
    os.mkdir('test/dog')
 
    for file_train in os.listdir('dogs-vs-cats/train'):
        if 'dog' in file_train:
            shutil.move(f'dogs-vs-cats/train/{file_train}', f'train/dog')
        elif 'cat' in file_train:
            shutil.move(f'dogs-vs-cats/train/{file_train}', f'train/cat')
        else:
            pass

    for file_train in os.listdir('train/dog')[:500]:
        shutil.move(f'train/dog/{file_train}', f'test/dog')

    for file_train in os.listdir('train/cat')[:500]:
        shutil.move(f'train/cat/{file_train}', f'test/cat')

