"""Classifier pipeline, grouped cross-validation, and metrics."""

import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from csc74030_zoi.features import (haralick_features, box_count_features, chain_code_features, 
                               areas_features, lbp_features)

FUNCS = [haralick_features, box_count_features, chain_code_features, 
         areas_features, lbp_features]

def count_class(df, label="label"):
    counts = df[[label]].value_counts()
    percent = df[[label]].value_counts(normalize=True) * 100
    return pd.DataFrame({"count": counts, "percent": percent.round(1)})

def extract_features(image, funcs):
    features = {}
    for func in funcs:
        features = features | func(image)
    return features

def prepare_features_for_images(df, funcs=FUNCS, pyramid=None, verbose=True):
    """Expects columns: label, roi, image; can take a pyramid function; returns train_X, labels"""
    all_features = []
    labels = []
    for row in df.itertuples():
        if verbose:
            print(f"Processing {row.roi} with label {row.label}")
        if pyramid is not None:
            features = {}

            pyr = pyramid(row.image)
            for i, level in enumerate(pyr):
                roi_features = extract_features(level, funcs)
                features |= {f"{key}_L{i}": value for key, value in roi_features.items()}
        else:
            features = extract_features(row.image, funcs)
        all_features.append(features)
        labels.append(row.label)
    return pd.DataFrame(all_features), np.array(labels)

# The following code is based on https://gogulilango.com/software/texture-recognition
def train_features(train_X, labels, random_state=5, max_iter=1000, verbose=True):
    """Takes train_X, labels from prepare_features_for_images"""
    if verbose:
        print(f"Features: {train_X.shape}, Labels: {len(labels)}")
    svm = make_pipeline(StandardScaler(),
                        LinearSVC(random_state=random_state,
                                  max_iter=max_iter))
    if verbose:
        print("Fitting data and labels to model")
    svm.fit(train_X, labels)
    return svm

# Suggested by Claude Sonnet 5.5
def roi_group(df):
    """Uses roi column to split into groups"""
    return df["roi"].str.rsplit("-y", n=1).str[0].to_numpy()

