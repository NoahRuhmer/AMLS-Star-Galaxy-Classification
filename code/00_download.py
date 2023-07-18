import os.path
import requests

base_url = "https://data.sdss.org/sas/dr17/eboss/"
url_frame = "photoObj/frames/"
url_coords = "sweeps/dr13_final/"

if not os.path.exists("../data"):
    os.mkdir("../data")
os.chdir("../data")


def request_and_write(url, filename):
    path_name = "/".join(filename.split('/')[:-1])
    if not os.path.exists(path_name):
        os.makedirs(path_name)
    with requests.get(url=url, stream=True) as request:
        request.raise_for_status()
        with open(file=filename, mode='wb') as file:
            for chunk in request.iter_content(chunk_size=4096):
                file.write(chunk)


def download_csv(rerun, run, camcol):
    csv_url = f"{base_url}{url_coords}{rerun}/calibObj-{str(run).zfill(6)}-{camcol}"
    gal_url = csv_url + "-gal.fits.gz"
    star_url = csv_url + "-star.fits.gz"
    filename = f"rerun_{rerun}/run_{run}/camcol_{camcol}/"
    request_and_write(gal_url, filename + "gal.fits.gz")
    request_and_write(star_url, filename + "star.fits.gz")


def download_frame(rerun, run, camcol, field):
    start_url = f"{base_url}{url_frame}{rerun}/{run}/{camcol}/frame-"
    end_url = f"-{str(run).zfill(6)}-{camcol}-{str(field).zfill(4)}"
    start_filename = f"rerun_{rerun}/run_{run}/camcol_{camcol}/field_{field}_"

    for band_char in ["i", "r", "g", "z", "u", "irg"]:
        url = start_url + band_char + end_url
        filename = start_filename + band_char
        if band_char == "irg":
            request_and_write(url + ".jpg", filename + ".jpg")
        else:
            request_and_write(url + ".fits.bz2", filename + ".fits.bz2")


rerun = 301
run = 8162
camcol = 6
fields = [80, 103, 111, 120, 147, 174, 177, 214, 222, 228]

print(f"Downloading files for {len(fields)} fields:")
try:
    print(f"- Coordinate files...", end=" ")
    download_csv(rerun, run, camcol)
    print(f"done!")

    for field in fields:
        print(f"- Field {field}...", end=" ")
        download_frame(rerun, run, camcol, field)
        print(f"done!")
    print("File downloads successful.")
except:
    print("Downloads Failed. Please try again.")
