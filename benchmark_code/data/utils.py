"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from pypots.data.saving import save_dict_into_h5


def organize_and_save(data_dict, saving_dir):
    train = {
        "X": data_dict["train_X"],
        "X_ori": data_dict["train_X_ori"] if "train_X_ori" in data_dict.keys() else "",
        "y": data_dict["train_y"] if "train_y" in data_dict.keys() else "",
    }
    val = {
        "X": data_dict["val_X"],
        "X_ori": data_dict["val_X_ori"],
        "y": data_dict["val_y"] if "val_y" in data_dict.keys() else "",
    }
    test = {
        "X": data_dict["test_X"],
        "X_ori": data_dict["test_X_ori"],
        "y": data_dict["test_y"] if "test_y" in data_dict.keys() else "",
    }
    save_dict_into_h5(train, saving_dir, "train.h5")
    save_dict_into_h5(val, saving_dir, "val.h5")
    save_dict_into_h5(test, saving_dir, "test.h5")
    print("\n\n\n")
