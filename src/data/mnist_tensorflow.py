import tensorflow_datasets as tfds

def load_data():
    ds_builder = tfds.builder('mnist')
    ds_builder.download_and_prepare()

    train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=128, shuffle_files=True))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=10000))
    return train_ds, test_ds