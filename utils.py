from torch.utils.data import Dataset


def print_args(args, get_str=False):
    if "delimiter" in args:
        delimiter = args.delimiter
    elif "sep" in args:
        delimiter = args.sep
    else:
        delimiter = ";"
    print("###################################################################")
    print("args: ")
    keys = sorted(
        [
            a
            for a in dir(args)
            if not (
                a.startswith("__")
                or a.startswith("_")
                or a == "sep"
                or a == "delimiter"
        )
        ]
    )
    values = [getattr(args, key) for key in keys]
    if get_str:
        keys_str = delimiter.join([str(a) for a in keys])
        values_str = delimiter.join([str(a) for a in values])
        print(keys_str)
        print(values_str)
        return keys_str, values_str
    else:
        for key, value in zip(keys, values):
            print(key, ": ", value, flush=True)
    print("ARGS FINISHED", flush=True)
    print("######################################################")


# custom dataset class
class CustomDataset(Dataset):
    def __init__(self, images, labels=None, transforms=None):
        self.labels = labels
        self.images = images
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        data = self.images[index][:]

        if self.transforms:
            data = self.transforms(data)

        if self.y is not None:
            return (data, self.labels[index])
        else:
            return data

# train_data = CustomDataset(x_train, y_train, train_transforms)
