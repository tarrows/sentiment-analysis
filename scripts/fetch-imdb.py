import tensorflow as tf

NAME = 'aclImdb_v1.tar.gz'
URL = f'https://ai.stanford.edu/~amaas/data/sentiment/{NAME}'
SEED = 123

dataset = tf.keras.utils.get_file(fname=NAME, origin=URL, untar=True, cache_dir='./data', cache_subdir='')

print(dataset)
