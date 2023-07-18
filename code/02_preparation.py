import os
import numpy as np


def extract_target_patches_from_frame(frame, gals, stars, patch_len):
    assert patch_len % 2, "patch length should be odd"
    offset = patch_len // 2

    def extract_padded_patches(frame, data, offset):
        channels, max_height, max_width = frame.shape

        padded_frame = np.pad(frame, offset + 1, 'constant', constant_values=0)
        padded_frame = padded_frame[offset + 1:-offset - 1]

        patch_centers = np.rint(data).astype(int)
        patch_centers = patch_centers + offset + 1

        right_bottom = patch_centers + offset + 1
        left_top = patch_centers - offset

        patches = np.zeros((len(patch_centers), channels, patch_len, patch_len))
        for idx in range(len(patch_centers)):
            patches[idx] = padded_frame[:, left_top[idx, 1]:right_bottom[idx, 1], left_top[idx, 0]:right_bottom[idx, 0]]
        return patches

    gal_patches = np.moveaxis(np.array(extract_padded_patches(frame, gals, offset)), 1, 3)
    star_patches = np.moveaxis(np.array(extract_padded_patches(frame, stars, offset)), 1, 3)
    return gal_patches, star_patches


def save_target_patches(rerun, run, camcol, fields, patch_len):
    path = f"../aligned_data/rerun_{rerun}/run_{run}/camcol_{camcol}/"
    frames = np.load(path + "frames_aligned.npy", allow_pickle=True).item()
    gals = np.load(path + "target_gals.npy", allow_pickle=True).item()
    stars = np.load(path + "target_stars.npy", allow_pickle=True).item()

    gal_patches = {}
    star_patches = {}
    for key in fields:
        print(f"- Field {key}...", end=" ")
        key_str = str(key)
        if key_str in frames.keys():
            frame = frames[key_str][0]
            gal = gals[key_str]
            star = stars[key_str]
            gal_patches[key_str], star_patches[key_str] = extract_target_patches_from_frame(frame, gal, star, patch_len)
        print("done!")
    np.save(path + f"patches{patch_len}_gals", gal_patches)
    np.save(path + f"patches{patch_len}_stars", star_patches)


def get_ml_data(star_patches, gal_patches, fields, patch_size):
    data = np.zeros((0, patch_size, patch_size, 5))
    for key in fields:
        data = np.append(data, gal_patches[str(key)], axis=0)
    gal_count = len(data)
    for key in fields:
        data = np.append(data, star_patches[str(key)], axis=0)
    targets = np.ones(len(data))
    targets[:gal_count] = 0

    return data, targets


def save_ml_data(rerun, run, camcol, train_fields, val_fields, test_fields, patch_size, identifier):
    path = f"../aligned_data/rerun_{rerun}/run_{run}/camcol_{camcol}/"
    gal_patches = np.load(path + f"patches{patch_size}_gals.npy", allow_pickle=True).item()
    star_patches = np.load(path + f"patches{patch_size}_stars.npy", allow_pickle=True).item()

    train_data, train_targets = get_ml_data(gal_patches, star_patches, train_fields, patch_size)
    val_data, val_targets = get_ml_data(gal_patches, star_patches, val_fields, patch_size)
    test_data, test_targets = get_ml_data(gal_patches, star_patches, test_fields, patch_size)

    ml_data_path = f"../ml_data/{identifier}/"
    if not os.path.exists(ml_data_path):
        os.makedirs(ml_data_path)
    np.save(ml_data_path + "train_data", train_data)
    np.save(ml_data_path + "train_targets", train_targets)
    np.save(ml_data_path + "val_data", val_data)
    np.save(ml_data_path + "val_targets", val_targets)
    np.save(ml_data_path + "test_data", test_data)
    np.save(ml_data_path + "test_targets", test_targets)


rerun = 301
run = 8162
camcol = 6

patch_size = 25

# use this to use the large training data set
train_fields = [103, 111, 147, 174, 177, 214, 222]
identifier = f"patch_size{patch_size}_frames_10_ref_test"

# use this to use the small training data set
# train_fields = [174]
# identifier = f"patch_size{patch_size}_frames_3_ref_test"

val_fields = [120, 228]
test_fields = [80]

fields = train_fields + val_fields + test_fields
print(f"Building training, validation and test data sets.\n")
print(f"Training fields: {train_fields}")
print(f"Validation fields: {train_fields}")
print(f"Test fields: {train_fields}\n")
try:
    print(f"Extracting {patch_size}x{patch_size} patches from {len(fields)} fields:")
    save_target_patches(rerun, run, camcol, fields, patch_size)
    print(f"All patches extracted!\n")

    print("Extracting target data...", end=" ")
    save_ml_data(rerun, run, camcol, train_fields, val_fields, test_fields, patch_size, identifier)
    print(f"done!\n")

    print("Machine Learning data preparation successful.")
except:
    print()
    print("Error in data preparation. Please try again.")
