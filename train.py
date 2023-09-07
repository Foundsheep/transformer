from layers import Transformer, CustomSchedule
from data_prep import DataLoader

from pathlib import Path


def train(d_k, d_v, d_model, vocab_size, h):

    # data preparation
    dl = DataLoader()
    ko_dict, ko_dict_inv, en_dict, en_dict_inv, df_train, df_val = dl.load_dataset()
    if not dl.save_path_ko_dict.exists():
        dl.save_data()

    # model generation
    transformer_model = Transformer(d_k=d_k, d_v=d_v, d_model=d_model, vocab_size=vocab_size, h=h)

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
    transformer_model.fit(df_train, validation_data=df_val, epochs=epochs)
