import os
import bz2
import numpy as np
import shutil

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.units import deg

from reproject import reproject_exact


def load_frames(rerun, run, camcol, fields):
    path = f"../data/rerun_{rerun}/run_{run}/camcol_{camcol}/"
    assert os.path.exists(path), "Path to load from does not exist"

    aligned_path = f"../aligned_data/rerun_{rerun}/run_{run}/camcol_{camcol}/"
    if not os.path.exists(aligned_path):
        os.makedirs(aligned_path)

    files = os.listdir(path)
    assert files, "No files in target path found"

    frames_data_wcs = {}
    for file in files:
        if file.startswith(f"field_") and file.endswith(".fits.bz2"):
            field = file[6:-11]
            if field not in frames_data_wcs.keys():
                frames_data_wcs[field] = {}
            bz2_file = bz2.BZ2File(path + file)
            hdul = fits.open(bz2_file)
            frames_data_wcs[field][file[-10]] = (np.array(hdul[0].data), WCS(hdul[0].header))
        if file.endswith("fits.gz"):
            shutil.copyfile(path + file, aligned_path + file)

    assert len(frames_data_wcs) != 0, f"Files for field {fields} not found in {path}"
    return frames_data_wcs


def align_frame(frames):
    ref_band = "r"
    ref_data, ref_wcs = frames[ref_band]
    aligned_frame = np.zeros((5, *ref_data.shape))
    for idx, band in enumerate(["i", "r", "g", "z", "u"]):
        if band == ref_band:
            aligned_frame[idx], _ = frames[band]
        aligned_frame[idx] = reproject_exact(frames[band], ref_wcs, shape_out=ref_data.shape, return_footprint=False)
    return np.nan_to_num(aligned_frame), ref_wcs


def align_and_save_frames(rerun, run, camcol, fields):
    path = f"../aligned_data/rerun_{rerun}/run_{run}/camcol_{camcol}/"
    if not os.path.exists(path):
        os.makedirs(path)
    print(f"- Loading frames...", end=" ")
    frames = load_frames(rerun, run, camcol, fields)
    print("done!")
    aligned_frames = {}
    for key in frames:
        print(f"- Aligning field {key}...", end=" ")
        aligned_frames[key] = align_frame(frames[key])
        print("done!")

    np.save(f"{path}frames_aligned", aligned_frames)


def load_star_gal_data(path):
    assert os.path.exists(path), "Path to load from does not exist"
    files = os.listdir(path)
    assert files, "No files in target path found"

    frames = {}
    stars = []
    gals = []
    for file in files:
        if file == "frames_aligned.npy":
            frames = np.load(path + file, allow_pickle=True).item()
        if file == "gal.fits.gz":
            gals = fits.open(path + file)[1]
        if file == "star.fits.gz":
            stars = fits.open(path + file)[1]
    return frames, gals, stars


def extract_field_coords(data, wcs, field, height_ref, width_ref):
    pixels = []
    for entry in data:
        # look at pixels to next fields as they are overlapping
        if field - 1 <= entry["FIELD"] <= field + 1:
            coords = SkyCoord(entry["RA"], entry["DEC"], unit=deg)
            width, height = wcs.world_to_pixel(coords)

            # only use pixels within this fields frame
            if -0.5 < width < width_ref + 0.5 and -0.5 < height < height_ref + 0.5:
                pixels.append(np.array([width, height]))
    return pixels


def target_coords_for_field(frame, field, gals, stars):
    wcs = frame[1]
    _, height_ref, width_ref = frame[0].shape
    gal_pixels = extract_field_coords(gals.data, wcs, field, height_ref, width_ref)
    star_pixels = extract_field_coords(stars.data, wcs, field, height_ref, width_ref)
    return np.array(gal_pixels), np.array(star_pixels)


def save_target_coords(rerun, run, camcol):
    path = f"../aligned_data/rerun_{rerun}/run_{run}/camcol_{camcol}/"
    frames, gals, stars = load_star_gal_data(path)
    gal_targets = {}
    star_targets = {}
    for key in frames.keys():
        gal_targets[key], star_targets[key] = target_coords_for_field(frames[key], int(key), gals, stars)
    np.save(f"{path}target_gals", gal_targets)
    np.save(f"{path}target_stars", star_targets)


rerun = 301
run = 8162
camcol = 6
fields = [80, 103, 111, 120, 147, 174, 177, 214, 222, 228]

try:
    print(f"Aligning frames for {len(fields)} fields:")
    align_and_save_frames(rerun, run, camcol, fields)
    print(f"All frames aligned!")

    print("Extracting galaxy and star coordinates...", end=" ")
    save_target_coords(rerun, run, camcol)
    print(f"done!")

    print("Field alignments and galaxy/star coordinate extraction successful.")
except:
    print("Error in alignment or coordinate extraction. Please try again.")
