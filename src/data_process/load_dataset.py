from datasets import load_dataset


def load_dataset_from_local(path: str):
    dataset = load_dataset(path)
    return dataset

def load_dataset_from_hf(name: str, **kwargs):
    """
    Load a dataset from Hugging Face datasets hub.
    
    :param name: Name of the dataset on Hugging Face
    :param kwargs: Additional arguments to pass to load_dataset, including subset
    :return: The loaded dataset
    """
    dataset = load_dataset(name, **kwargs)
    return dataset


if __name__ == "__main__":
    # Specify both subset and split for MMLU dataset
    params = {
        "name": "all",  # This is the subset for MMLU
        "split": "test"  # Specify a valid split: 'train', 'test', or 'validation'
    }
    dataset = load_dataset_from_hf("cais/mmlu", **params)
    print(dataset)


