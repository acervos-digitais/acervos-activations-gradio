import matplotlib.cm as cm

import numpy as np

from PIL import Image as PImage
from scipy.interpolate import RBFInterpolator

def scale_2d_array(array, size, sampling=PImage.Resampling.BILINEAR):
  if len(array.shape) == 1:
    dim = int(array.shape[0] ** 0.5)
    array = array.reshape(dim, dim)
  return np.array(PImage.fromarray(array).resize(size, resample=sampling))

def scale_2d_array_rbf(array, size, kernel="thin_plate_spline"):
  x0 = np.linspace(0, size[0], array.shape[0])
  y0 = np.linspace(0, size[1], array.shape[0])
  xy0 = np.array([[x,y] for y in y0 for x in x0])
  z0 = array.reshape(-1)
  rbf = RBFInterpolator(xy0, z0, kernel=kernel)
  X1 = np.arange(0, size[0])
  Y1 = np.arange(0, size[1])
  XY1 = np.asarray(np.meshgrid(X1, Y1, indexing="xy"))
  XY1_flat = XY1.reshape(2, -1).T
  Z1_flat = rbf(XY1_flat)
  return Z1_flat.reshape(size[1], size[0])

def mask_image(img, mask, sampling=PImage.Resampling.BILINEAR):
  img_np = np.array(img)

  if len(mask.shape) < 2 or mask.shape[0] != img_np.shape[0] or mask.shape[1] != img_np.shape[1]:
    mask = scale_2d_array(mask, img.size, sampling=sampling)

  if len(mask.shape) == 2:
    mask = mask[:, :, None]

  return PImage.fromarray((mask * img_np).astype(np.uint8))

# map := [ 'viridis', 'plasma', 'inferno', 'magma' ]
def heatmap_image(data, *, size=None, cmap="inferno", sampling=PImage.Resampling.BILINEAR):
  if size:
    data = scale_2d_array(data, size, sampling=sampling)
  map_fun_np = np.vectorize(cm.get_cmap(cmap))
  rgba_np = 255 * np.stack(map_fun_np(data)[:3], axis=-1)
  himg = PImage.fromarray(rgba_np.astype(np.uint8))
  if size:
    himg = himg.resize(size)
  return himg

def heatmap_image_rbf(data, *, size=None, cmap="inferno", kernel="thin_plate_spline"):
  if size:
    data = scale_2d_array_rbf(data, size, kernel=kernel)
  map_fun_np = np.vectorize(cm.get_cmap(cmap))
  rgba_np = 255 * np.stack(map_fun_np(data)[:3], axis=-1)
  himg = PImage.fromarray(rgba_np.astype(np.uint8))
  if size:
    himg = himg.resize(size)
  return himg
