from layers import Transformer, CustomSchedule
from data_prep import DataLoader, convert_to_ds, make_batches
from matplotlib import pyplot as plt

from pathlib import Path


def train(d_k, d_v, d_model, vocab_size, h, N, batch_size=64, is_plot=True):

    # data preparation
    dl = DataLoader()
    ko_dict, ko_dict_inv, en_dict, en_dict_inv, df_train, df_val = dl.load_dataset()
    if not dl.save_path_ko_dict.exists():
        dl.save_data()

    ds_train = convert_to_ds(df_train["ko_tokenized"], df_train["en_tokenized"])
    ds_train = make_batches(ds_train)
    ds_val = convert_to_ds(df_val["ko_tokenized"], df_val["en_tokenized"])
    ds_val = make_batches(ds_val)

    # model generation
    transformer_model = Transformer(d_k=d_k, d_v=d_v, d_model=d_model, vocab_size=vocab_size, h=h, N=N)

    # model configuration
    custom_scheduler = CustomSchedule(d_model=d_model)
    optimizer = "adam"
    loss = "cross_entropy"
    metrics = ["accuracy"]
    epochs = 10
    transformer_model.compile(optimizer=optimizer,
                              loss=loss,
                              metrics=metrics)

    # train model
    history = transformer_model.fit(ds_train, validation_data=ds_val, epochs=epochs)

    if is_plot:
        pass


if __name__ == "__main__":
    d_k = 64
    d_v = 64
    b_size = 64
    s_size = 100
    vocab_size = 10000
    d_model = 512
    h = 8
    N = 6

    train(d_k=d_k, d_v=d_v, d_model=d_model, vocab_size=vocab_size, h=h, N=N)
