"""Texture and shape features: Haralick, box counting, chain code, areas, LBP."""

import numpy as np
import cv as cv2


def glcm(image, directions=[0, 45, 90, 135], d=1):
    """Reduces to 16"""

    if len(directions) == 0 or not set(directions).issubset(set([0, 45, 90, 135])):
        raise ValueError("Invalid direction. Must be at least one of [0, 45, 90, 135]")
    # Convert to grayscale if color
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image // 16
    glcms = {direction: np.zeros([16, 16]) for direction in directions}
    height, width = image.shape[:2]
    if d <= 0 or d >= height or d >= width:
        raise ValueError(f"d must be greater than 0 but less than {height} or {width}")
    for i in range(height):
        for j in range(width):
            if 0 in directions and j+d < width:
                glcms[0][image[i,j], image[i,j+d]] += 1
            if 45 in directions and 0 <= i-d and j+d < width:
                glcms[45][image[i,j], image[i-d,j+d]] += 1
            if 90 in directions and 0 <= i-d:
                glcms[90][image[i,j], image[i-d,j]] += 1
            if 135 in directions and 0 <= i-d and 0 <= j-d:
                glcms[135][image[i,j], image[i-d,j-d]] += 1
    normalized_tables = {}
    for direction in directions:
        symmetric_glcm = glcms[direction] + glcms[direction].T
        normalized_tables[direction] = symmetric_glcm / symmetric_glcm.sum()
    return normalized_tables

def haralick_features(image, k=2, n=1, directions=[0, 45, 90, 135], d=1):
    nt = glcm(image, directions=directions, d=d)

    features = {}
    stack = np.stack(list(nt.values()))
    levels = np.indices((16,16))
    # (avg) energy: p ** 2
    features["haralick_energy"] = np.square(stack).sum() /  len(nt)
    # (avg) entropy: p * log p
    nonzero_stack = stack[stack > 0]
    features["haralick_entropy"] = (-nonzero_stack * np.log(nonzero_stack)).sum() / len(nt)
    # (avg) contrast: (i-j) ** k * p ** n where i and j are the levels
    features["haralick_contrast"] = (((levels[0] - levels[1]) ** k) * (stack ** n)).sum() / len(nt)
    # (a single value out of all values in all directions) max probability: max(p)
    features["haralick_max"] = np.max(stack)
    # homogeneity: p / (1+|i-j|)
    features["haralick_homogeneity"] = (stack / (1+np.absolute(levels[0]-levels[1]))).sum() / len(nt)
    return features

def N(image, s):
    height, width = image.shape[:2]
    hs, ws = height // s, width // s
    n_s = 0
    for h in range(hs):
        for w in range(ws):
            n_s += np.any(image[h*s:h*s+s, w*s:w*s+s] == 255)
    return n_s

# box counting
def box_count(image):
    """image: binary image"""

    image = get_binary_image(image)
    height, width = image.shape[:2]
    gcd = math.gcd(height, width)

    d = []
    n = {}
    ss = [s for s in range(2, gcd) if height % s == 0 and width % s == 0 and s > 1 and s < height and s < width]
    if len(ss) < 2:
        return np.nan
    
    for s in ss:
        n[s] = N(image, s)

    s = ss[0]
    ns = n[s]
    for s1 in ss[1:]:
        if ns == 0 or n[s] == 0:
            continue
        # recommended by claude sonnet 5.5 to switch order of denominator to make positive
        d.append((math.log(ns) - math.log(n[s1])) / (math.log(s1) - math.log(s)))
        s = s1
        ns = n[s1]

    return np.mean(d) if len(d) > 0 else np.nan

def box_count_features(image):
    return {"box_count": box_count(image)}
    
def get_chain_code(image):
    """image must be binary"""
    image = get_binary_image(image)
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return contours, cv2.drawContours(canvas, contours, -1, (0,255,0), 1)

def bin_directions(points):
    values = np.zeros(8)
    point = points[:1]
    for next_point in np.concatenate((points[1:], point)):
        if np.all((next_point - point) == np.array([[-1, 0]])):
            values[0] += 1
        elif np.all((next_point - point) == np.array([[-1, 1]])):
            values[1] += 1
        elif np.all((next_point - point) == np.array([[0, 1]])):
            values[2] += 1
        elif np.all((next_point - point) == np.array([[1, 1]])):
            values[3] += 1
        elif np.all((next_point - point) == np.array([[1, 0]])):
            values[4] += 1
        elif np.all((next_point - point) == np.array([[1, -1]])):
            values[5] += 1
        elif np.all((next_point - point) == np.array([[0, -1]])):
            values[6] += 1
        elif np.all((next_point - point) == np.array([[-1, -1]])):
            values[7] += 1
        else:
            raise ValueError(f"Difference greater than 1: {point} - {next_point}")
        point = next_point
    return values

def chain_code_features(image):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, 
                                            cv2.CHAIN_APPROX_NONE)
    values = np.zeros(8)
    for points in contours:
        if len(points) < 2:
            continue
        v = bin_directions(points)
        values = values + v
    values = np.zeros(8) if sum(values) == 0 else values / sum(values)
    return {f"chain_code_{i}": v for i, v in enumerate(values)}

def get_areas(image):
    """image must be binary"""

    image = get_binary_image(image)
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, 
                                    cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return 0, 0, 0

    num_of_contours = len(contours)
    net_area = np.count_nonzero(image == 255)
    convex_areas = np.zeros(num_of_contours)
    perimeters = np.zeros(num_of_contours)

    for i in range(len(contours)):
        hull = cv2.convexHull(contours[i])
        convex_areas[i] = cv2.contourArea(hull)
        perimeters[i] = cv2.arcLength(hull, closed=True)

    return net_area, np.mean(convex_areas), np.mean(perimeters)

def areas_features(image):
    net_area, convex_area, perimeter = get_areas(image)
    return {"net_area": net_area, "convex_area": convex_area, "perimeter": perimeter}

def lbp(image):
    # we'll do a simple version where p = 8 neighbors
    r = 1
    # Convert to grayscale if color
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = image.shape[:2]
    codes = np.zeros([height-2, width-2])
    for i in range(1, height-1):
        for j in range(1, width-1):
            # we should reorder for the weights
            neighbors = image[i-r:i+r+1,j-r:j+r+1].ravel()
            neighbors = neighbors[[5, 2, 1, 0, 3, 6, 7, 8]]
            # find only neighbors greater than or equal to the center pixel
            binary = neighbors >= image[i,j]
            # sum the weights that based on the order
            codes[i-1,j-1] = sum((2 ** np.arange(8)) * binary)
    return codes.astype(int)

def lbp_features(image):
    codes = lbp(image)
    counts, _ = np.histogram(codes, bins=256, range=(0,256))
    hist = counts / codes.size
    return {f"lbp_{i}": val for i, val in enumerate(hist)}
