# import pandas as pd
# import tensorflow as tf
# from typing import List
# from transformers import BertConfig, BertTokenizer, TFBertForSequenceClassification, InputExample, InputFeatures
from transformers import pipeline

# BATCH_SIZE = 30000
# SEED = 123


# def to_dataframe(dataset):
#     for i in dataset.take(1):
#         feature = i[0].numpy()
#         label = i[1].numpy()

#     df = pd.DataFrame([feature, label]).T
#     df.columns = ['review', 'label']
#     df['review'] = df['review'].str.decode("utf-8")
#     return df


# def make_dataset(data_dir: str):
#     args = {'batch_size': BATCH_SIZE, 'validation_split':0.2, 'seed':SEED}
#     train = tf.keras.preprocessing.text_dataset_from_directory(data_dir, subset='training', **args)
#     test = tf.keras.preprocessing.text_dataset_from_directory(data_dir, subset='validation', **args)

#     train = to_dataframe(train)
#     test = to_dataframe(test)

#     return train, test


# def to_tf_dataset(examples: List[InputExample], tokenizer: BertTokenizer, max_length=128):
#     features = []

#     for e in examples:
#         # tokenizer.encode_plus is deprecated
#         input_dict = tokenizer(e.text_a, add_special_tokens=True, max_length=max_length, return_token_type_ids=True, return_attention_mask=True, padding='max_length', truncation=True)
#         features.append(InputFeatures(input_ids=input_dict['input_ids'], attention_mask=input_dict['attention_mask'], token_type_ids=input_dict['token_type_ids'], label=e.label))

#     def generator():
#         for feature in features:
#             yield ({'input_ids': feature.input_ids, 'attention_mask': feature.attention_mask, 'token_type_ids': feature.token_type_ids}, feature.label)

#     return tf.data.Dataset.from_generator(
#         generator,
#         ({'input_ids': tf.int32, 'attention_mask': tf.int32, 'token_type_ids': tf.int32}, tf.int64),
#         ({'input_ids': tf.TensorShape([None]), 'attention_mask': tf.TensorShape([None]), 'token_type_ids': tf.TensorShape([None])}, tf.TensorShape([]))
#     )


# def main():
#     config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True, output_attentions=True, use_cache=True)
#     model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', config=config)
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', config=config)

#     train, test = make_dataset('data/aclImdb/train')

#     print('--- dataset prepared ---')
#     print(train.head())
#     print(train.dtypes)

#     def to_input_example(x):
#         return InputExample(guid=None, text_a=x['review'], text_b=None, label=x['label'])

#     train = train.apply(to_input_example, axis=1)
#     test = test.apply(to_input_example, axis=1)

#     train = to_tf_dataset(list(train), tokenizer)
#     train = train.shuffle(100).batch(32).repeat(2)

#     test = to_tf_dataset(list(test), tokenizer)
#     test = test.batch(32)

#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')])
#     model.fit(train, epochs=2, validation_data=test)
#     model.save('my_model')
    # take3 = train.take(3)
    # print(list(take3.as_numpy_iterator()))


if __name__ == '__main__':
    # main()
    classifier = pipeline('sentiment-analysis')
    x = classifier('Awesome story that I have ever seen.')
    print(x)  # [{'label': 'POSITIVE', 'score': 0.9998598098754883}]
