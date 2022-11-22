import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow_text

# -- Download the dataset
examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                               with_info=True,
                               as_supervised=True)

train_examples, val_examples = examples['train'], examples['validation']

# -- Display some examples
for pt_examples, en_examples in train_examples.batch(3).take(1):
    print('> Examples in Portuguese:')
    for pt in pt_examples.numpy():
        print(pt.decode('utf-8'))
    print()

    print('> Examples in English:')
    for en in en_examples.numpy():
        print(en.decode('utf-8'))

# -- Set up the tokenizer

# -- Preprocessing steps needed
# 1. shuffle
# 2. batch
# 3. map()
#   3.1. tokenize
#   3.2. use only a fixed-length of the sentences
#   3.3. padding
# 4. prefetch