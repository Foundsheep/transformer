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

# -- Download and load the tokenizer
model_name = 'ted_hrlr_translate_pt_en_converter'
tf.keras.utils.get_file(
    f'{model_name}.zip',
    f'https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip',
    cache_dir='.', cache_subdir='', extract=True
)

tokenizers = tf.saved_model.load(model_name)

# -- Set the MAX_TOKEN number
# TODO : might be better to save it once in a config file
lengths = []
for pt_examples, en_examples in train_examples.batch(2048):
    pt_tokens = tokenizers.pt.tokenize(pt_examples)
    lengths.append(pt_tokens.row_lengths())

    en_tokens = tokenizers.en.tokenize(en_examples)
    lengths.append(en_tokens.row_legnths())

lengths = np.concatenate(lengths)
q3 = np.quantile(legnths, 0.75)
print(f"Tokenised dataset length max : {legnths.max()}")
print(f"Tokenised dataset length min : {legnths.min()}")
print(f"Tokenised dataset length mean : {legnths.mean()}")
print(f"Tokenised dataset length median : {np.median(lengths)}")
print(f"Tokenised dataset length MAX_TOKEN : {q3}")

MAX_TOKEN = int(q3)


# -- Preprocessing steps needed
# 1. shuffle
# 2. batch
# 3. map()
#   3.1. tokenize
#   3.2. use only a fixed-length of the sentences
#   3.3. padding
# 4. prefetch