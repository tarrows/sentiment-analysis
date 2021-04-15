import pandas as pd
import tensorflow as tf

BATCH_SIZE = 30000
SEED = 123

def to_dataframe(dataset):
    for i in dataset.take(1):
        feature = i[0].numpy()
        label = i[1].numpy()

    df = pd.DataFrame([feature, label]).T
    df.columns = ['review', 'label']
    df['review'] = df['review'].str.decode("utf-8")
    return df


def make_dataset(data_dir: str):
    train = tf.keras.preprocessing.text_dataset_from_directory(
        data_dir, batch_size=BATCH_SIZE, validation_split=0.2,
        subset='training', seed=SEED
    )

    test = tf.keras.preprocessing.text_dataset_from_directory(
        data_dir, batch_size=BATCH_SIZE, validation_split=0.2,
        subset='validation', seed=SEED
    )

    train = to_dataframe(train)
    test = to_dataframe(test)

    return train, test


if __name__ == '__main__':
    train, test = make_dataset('data/aclImdb/train')
    print(train.head())
