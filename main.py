# /usr/bin/python3
from transfer_functions import TF_LogSNRThr
from camera_params import Camera
from imageio import imread, imwrite
import numpy as np
import sys

cam = Camera['SonyA7r1']
bits = 8
snr_db = -10

tf = TF_LogSNRThr(bits, snr_db, np.array(cam.k).mean(), cam.std_read, cam.std_adc)

def transfer(im):
    # conv_factor = 1 / np.percentile(im, 99.9) * 0.999
    conv_factor = (2**bits-1) / np.amax(im)
    im = im * conv_factor
    im = tf.encode(im)
    im = (im * 255).astype(np.uint8)
    return im

if __name__ == "__main__":
    try:
        *rest, hdrim, ldrim = sys.argv
    except:
        print("usage: python main.py <inputfile> <outputfile>")
        sys.exit(0)

    im = imread(hdrim)
    imwrite(ldrim, transfer(im))
