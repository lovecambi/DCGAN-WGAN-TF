# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 17:24:35 2017

@author: fankai

modified from https://github.com/nivwusquorum/tf-adversarial/blob/master/Adversarial-LSUN.ipynb
"""

from PIL import Image
import io
import numpy as np
import lmdb
import scipy.misc

DB_PATH = './data/lsun/bedroom_train_lmdb'

def load_image(val):
    """LSUN images are stored as bytes of JPEG representation.
    This function converts those bytes into a a 3D tensor 
    of shape (64,64,3) in range [0.0, 1.0].
    """
    img = Image.open(io.BytesIO(val))
    rp = 64.0 / min(img.size)
    img = img.resize(np.rint(rp * np.array(img.size)).astype(np.int32), Image.BICUBIC)
    img = img.crop((0,0,64,64))
    img = np.array(img, dtype=np.float32) / 127.5 - 1
    return img


def iterate_images(start_idx=None):
    """Iterates over the images returns pairs of 
    (index, image_tensor). It is never the case that all the images
    are loaded into memory at the same time (hopefully, lmdb, please?).
    
    give it a start_idx, not to start from the beginning"""
    with lmdb.open(DB_PATH, map_size=1099511627776,
                    max_readers=100, readonly=True) as env:
        with env.begin(write=False) as txn:
            with txn.cursor() as cursor:
                for i, (key, val) in enumerate(cursor):
                    if start_idx is None or start_idx <= i:
                        yield i, load_image(val)

                        
def batched_images(start_idx=None, DISCRIMIN_BATCH=64):
    """Yields pairs (start_idx_of_next_batch, batch). 
    Every batch is of shape (DISCRIMIN_BATCH, 64, 64, 3)"""
    batch, next_idx = None, None
    for idx, image in iterate_images(start_idx):
        if batch is None:
            batch = np.empty((DISCRIMIN_BATCH, 64, 64, 3))
            next_idx = 0
        batch[next_idx] = image
        next_idx += 1
        if next_idx == DISCRIMIN_BATCH:
            yield idx + 1, batch
            batch = None

#idx, img = next(iterate_images(0))
#scipy.misc.imsave('{:0>7}'.format(str(idx))+".jpg", img)
