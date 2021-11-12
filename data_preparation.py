import os
import zipfile
from proj_pspnet.constants.constants import DATA_PATH

if __name__ == '__main__':
    # creating folder for our data
    os.mkdir(DATA_PATH + '/ade20k_sub')

    # extracting all data from zip
    with zipfile.ZipFile(DATA_PATH + '/archive.zip', 'r') as zip_ref:
        zip_ref.extractall(DATA_PATH + '/ade20k_sub')
