"""(De)noise filters"""

import numpy as np
from scipy import stats

ORDER_STATISTIC = ("median", "min", "max", "weighted_median", "midpoint",
                   "alpha_trimmed_mean")

def uniform(img, mean, variance):
  """Generate Uniform noise"""

  low = mean - np.sqrt(3 * variance)
  high = mean + np.sqrt(3 * variance)
  uniform_noise = np.random.uniform(low, high, img.shape).astype(np.float32)
  noisy_img = img.astype(np.float32) + uniform_noise
  noisy_img = np.rint(np.clip(noisy_img, 0, 255)).astype(np.uint8)
  return noisy_img

def gaussian(img, mean, variance):
  """Generate Gaussian noise"""

  gaussian_noise = np.random.normal(mean, np.sqrt(variance),
                                    img.shape).astype(np.float32)
  noisy_img = img.astype(np.float32) + gaussian_noise
  noisy_img = np.rint(np.clip(noisy_img, 0, 255)).astype(np.uint8)
  return noisy_img

def snp(img, mean, variance):
  """Generate Salt and Pepper noise"""

  p_b = mean / 255
  p_a = np.where(mean != 0,
                 (variance - 255 * mean + 2 * (mean ** 2) - (mean ** 3) / 255) / (mean ** 2),
                 0)
  r = np.random.rand(*img.shape)
  pepper = r < p_a
  salt = r < p_a + p_b
  noisy_img = img.copy()
  noisy_img[pepper] = 0
  noisy_img[salt] = 255
  noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
  return noisy_img

def salt_and_pepper(img, density = 0.2):
  """Generate Salt and Pepper noise"""

  r = np.random.rand(*img.shape)
  pepper = r < density / 2
  salt = r >  1 - density /2
  noisy_img = img.copy()
  noisy_img[pepper] = 0
  noisy_img[salt] = 255
  noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
  return noisy_img

def rayleigh(img, mean, variance):
  """Generate Rayleigh noise"""

  b = 4 * variance / (4 - np.pi)
  a = mean - np.sqrt(np.pi * b / 4)
  scale = np.sqrt(b / 2)
  rayleigh_noise = a + np.random.rayleigh(scale, img.shape).astype(np.float32)
  noisy_img = img.astype(np.float32) + rayleigh_noise
  noisy_img = np.rint(np.clip(noisy_img, 0, 255)).astype(np.uint8)
  return noisy_img

def lognormal(img, mean, variance):
  """Generate lognormal noise"""

  sigma_squared = np.log(1 + variance / mean ** 2)
  sigma = np.sqrt(sigma_squared)
  mu = np.log(mean) - sigma_squared / 2
  lognormal_noise = np.random.lognormal(mu, sigma, img.shape).astype(np.float32)
  noisy_img = img.astype(np.float32) + lognormal_noise
  noisy_img = np.rint(np.clip(noisy_img, 0, 255)).astype(np.uint8)
  return noisy_img

# 5.2 Estimating noise parameters

def estimate_parameters(orig, noisy, tol = 4):
    """This is not accurate"""
    noise = noisy.astype(np.float32) - orig.astype(np.float32)
    pixels = noisy.ravel().astype(float)
    plt.figure(figsize = (6, 3.5))
    bins = np.arange(noise.min() - 0.5, noise.max() + 1.5, 1)
    values, edges, _ = plt.hist(noise.ravel(), bins = bins)
    pepper = np.mean(values == 0)
    salt = np.mean(values == 255)
    other = values[(values > 0) & (values < 255)]

    midpoints = (edges[:-1] + edges[1:]) / 2
    mean = np.average(midpoints, weights = values)
    variance = np.average((midpoints - mean) ** 2, weights = values)

    possible_types = []
    if other.size == 0:
        possible_types.append("salt-and-pepper")
    else:
        margin = 3 * other.std() + 3
        if (pepper > 0.002 and other.min() > margin) or (salt > 0.002 and 255 - other.max() > margin):
            possible_types.append("salt-and-pepper")

    skew = stats.skew(values)
    n = pixels.size
    skew, kurt = stats.skew(pixels), stats.kurtosis(pixels)
    se_s, se_k = np.sqrt(6 / n), np.sqrt(24 / n)
    if abs(skew) < tol * se_s:
        if abs(kurt + 1.2) < tol * se_k:
            possible_types.append("uniform")
        if abs(kurt) < tol * se_k:
            possible_types.append("gaussian")
    if abs(skew - 0.63) < tol * se_s and abs(kurt - 0.24) < tol * se_k:
        possible_types.append("rayleigh")

    if not possible_types:
        possible_types.append("unknown")
    return mean, variance, possible_types

# 5.3 Apply a class of order-statistics(median, max/min/weighted median,
# midpoint, alpha-trimmed mean) filters on generated noisy images

def alpha_trimmed_mean(x, d):
  x = np.sort(x)
  return x[d // 2: len(x) - d // 2].mean()

def midpoint(x):
  return (np.max(x) + np.min(x))/ 2

def weighted_median(x, w):
  """Assume x and w are flat, will error with negative values"""

  half = np.sum(w) / 2.0
  i = np.argsort(x)
  _x = x[i]
  _w = w[i]
  running = np.cumsum(_w)
  for j, v in enumerate(running):
    if v >= half:
      return _x[j]

def os_filter(img, k, order_statistic, weights=None, d=None):
  if k % 2 == 0:
    raise ValueError("k must be odd")

  if order_statistic not in ORDER_STATISTIC:
    raise ValueError(f"No such order statistic filter called {order_statistic}")
  elif order_statistic in ("median", "min", "max"):
    os_func = getattr(np, order_statistic)
  else:
    os_func = globals()[order_statistic]

  if order_statistic == "weighted_median":
    if weights is None or (k, k) != weights.shape:
      raise ValueError("weighted_median must have k * k weights")
    else:
      weights = weights.flatten()

  if order_statistic == "alpha_trimmed_mean" and \
    (d is None or (d % 2 != 0) or not (0 <= d  < (k ** 2))):
    raise ValueError("alpha_trimmed_mean must have an even d > 0")

  img = img.astype(np.float64)
  size = k // 2
  padded = np.pad(img, size, mode = "reflect")
  out = np.zeros_like(img)

  for i in range(img.shape[0]):
     for j in range(img.shape[1]):
        window = padded[i:i+k, j:j+k]
        flat_window = window.flatten()
        if order_statistic == "weighted_median":
          out[i, j] = os_func(flat_window, weights)
        elif order_statistic == "alpha_trimmed_mean":
          out[i, j] = os_func(flat_window, d)
        else:
          out[i, j] = os_func(flat_window)
  return np.rint(np.clip(out, 0, 255)).astype(np.uint8)

# 5.4 Adaptive local noise reduction


def adaptive_local_noise_reduction(img, var_h, k):
  """var_h: s_h ** 2
  Uses some flat patch
  var_h = patch.var(ddof = 1)
  adaptive_local_noise_reduction(image, var_h, 3)
  """
  if k % 2 == 0:
    raise ValueError("k must be odd")
  if var_h == 0:
    return img
  img = img.astype(np.float64)
  size = k // 2
  padded = np.pad(img, size, mode = "reflect")
  out = np.zeros_like(img)
  for i in range(img.shape[0]):
    for j in range(img.shape[1]):
      window = padded[i:i+k,j:j+k]
      m_l = window.mean()
      var_l = window.var(ddof = 1)  # s_l ** 2
      if var_l == var_h:
        out[i,j] = m_l
      elif var_l > var_h:
        out[i,j] = window[k // 2, k // 2] - (var_h / var_l) * (window[k // 2, k // 2] - m_l)
      else:
        out[i,j] = window[k // 2, k // 2] # pick original pixel
  return np.rint(np.clip(out, 0, 255)).astype(np.uint8)

# 5.5 Sharpening method using order-statistics (median, max/min/weighted median,
# midpoint, alpha-trimmed mean) filters

def os_filter_any(img, k, order_statistic, weights=None, d=None):
  if img.ndim == 2:
    return os_filter(img, k, order_statistic, weights=weights, d=d)
  return np.stack([os_filter(img[..., c], k, order_statistic, weights=weights, d=d)
                   for c in range(img.shape[2])], axis = -1)

def sharpen(img, k, a, order_statistic, weights=None, d=None):
  img = img.astype(np.float64)
  mask = img - os_filter_any(img, k, order_statistic, weights=weights, d=d)
  return np.rint(np.clip(img + a * mask, 0, 255)).astype(np.uint8)
