import os
import numpy as np
from scipy.ndimage import gaussian_filter


def load_data(identifier):
    ml_data_path = f"../ml_data/{identifier}/"
    train = np.load(ml_data_path + "train_data.npy", allow_pickle=True), \
        np.load(ml_data_path + "train_targets.npy", allow_pickle=True)
    val = np.load(ml_data_path + "val_data.npy", allow_pickle=True), \
        np.load(ml_data_path + "val_targets.npy", allow_pickle=True)
    test = np.load(ml_data_path + "test_data.npy", allow_pickle=True), \
        np.load(ml_data_path + "test_targets.npy", allow_pickle=True)
    return train, val, test


def save_augmented_data(train_augmented, val, test, identifier, aug_identifier):
    ml_data_path = f"../ml_data/{identifier}_{aug_identifier}/"
    if not os.path.exists(ml_data_path):
        os.makedirs(ml_data_path)

    train_data, train_targets = train_augmented
    val_data, val_targets = val
    test_data, test_targets = test

    np.save(ml_data_path + "train_data", train_data)
    np.save(ml_data_path + "train_targets", train_targets)
    np.save(ml_data_path + "val_data", val_data)
    np.save(ml_data_path + "val_targets", val_targets)
    np.save(ml_data_path + "test_data", test_data)
    np.save(ml_data_path + "test_targets", test_targets)


def min_max_norm(train, val, test):
    train_data, train_target = train
    val_data, val_target = val
    test_data, test_target = test
    mins = train_data.min(axis=(0, 1, 2))
    maxs = train_data.max(axis=(0, 1, 2))

    def rescale(data, mins, maxs):
        return (data - mins) / (maxs - mins)

    return (rescale(train_data, mins, maxs), train_target), \
        (rescale(val_data, mins, maxs), val_target), \
        (rescale(test_data, mins, maxs), test_target)


def rotations(data):
    patches, targets = data

    pos_rot = np.rot90(patches, k=1, axes=(1, 2))
    half_rot = np.rot90(patches, k=2, axes=(1, 2))
    neg_rot = np.rot90(patches, k=3, axes=(1, 2))

    new_data = np.concatenate((patches, pos_rot, neg_rot, half_rot))
    new_targets = np.concatenate((targets, targets, targets, targets))
    return new_data, new_targets


def reflections(data):
    patches, targets = data

    side_flip = np.flip(patches, axis=1)
    diag_flip = np.flip(patches, axis=(1, 2))

    new_data = np.concatenate((patches, side_flip, diag_flip))
    new_targets = np.concatenate((targets, targets, targets))
    return new_data, new_targets


def gaussian_blur(data):
    patches, targets = data
    blur_patches = gaussian_filter(patches, sigma=(0, 1, 1, 0))

    new_data = np.concatenate((patches, blur_patches))
    new_targets = np.concatenate((targets, targets))
    return new_data, new_targets


def brightness_shift(data, shift_intensity=0.1):
    patches, targets = data
    noise = np.random.randn(len(patches) * 5) * shift_intensity
    shifted_patches = patches + noise.reshape((len(patches), 1, 1, 5))

    new_data = np.concatenate((patches, shifted_patches))
    new_targets = np.concatenate((targets, targets))
    return new_data, new_targets


def augment_data(data):
    print(f"- Rotating...", end=" ")
    data = rotations(data)
    print("done!")
    print(f"- Reflecting...", end=" ")
    data = reflections(data)
    print("done!")
    print(f"- Shifting brightness...", end=" ")
    data = brightness_shift(data)
    print("done!")
    print(f"- Blurring...", end=" ")
    data = gaussian_blur(data)
    print("done!")
    return data


patch_size = 25

# identifier = f"patch_size{patch_size}_frames_3_ref_test"
identifier = f"patch_size{patch_size}_frames_10_ref_test"

train, val, test = load_data(identifier)

try:
    print(f"Augmenting training data:")
    augmented_train = augment_data(train)
    save_augmented_data(augmented_train, val, test, identifier, "augmented")
    print("Augmented training data successfully.")
except:
    print("Error in data augmentation. Please try again.")
