import torch
from captum.attr import (
    FeaturePermutation,
    GradientShap,
    IntegratedGradients,
    Occlusion,
    ShapleyValueSampling,
)
import numpy as np

def normalize(tensor, eps=1.0e-7):
    tensor -= np.min()
    tensor /= np.max() + eps
    return tensor


# Perturbation methods:



class FO:
    def __init__(self, f):
        self.f = f

    def attribute(self, X, normalize=True):
        explainer = Occlusion(forward_func=self.f)
        baseline = np.mean(X,axis=1)  # The baseline is chosen to be the average value for each feature
        attr = explainer.attribute(X, sliding_window_shapes=(1,), baselines=baseline)
        if normalize:
            attr = normal(npabs(attr))  # The absolute value of the FO attribution gives the feature importance
        return attr


class FP:
    def __init__(self, f):
        self.f = f

    def attribute(self, X, normalize=True):
        explainer = FeaturePermutation(forward_func=self.f)
        attr = explainer.attribute(X)
        if normalize:
            attr = normal(torch.abs(attr))  # The absolute value of the FP attribution gives the feature importance
        return attr


# Integrated Gradient:


class IG:
    def __init__(self, f):
        self.f = f

    def attribute(self, X, normalize=True):
        explainer = IntegratedGradients(forward_func=self.f)
        baseline = X * 0  # The baseline is chosen to be zero for all features
        attr = explainer.attribute(X, baselines=baseline)
        if normalize:
            attr = normal(torch.abs(attr))  # The absolute value of the IG attribution gives the feature importance
        return attr


# Shapley methods:


class GradShap:
    def __init__(self, f):
        self.f = f

    def attribute(self, X, normalize=True):
        explainer = GradientShap(forward_func=self.f, multiply_by_inputs=False)
        attr = explainer.attribute(X, baselines=torch.cat([0 * X, 1 * X]))
        if normalize:
            attr = normal(
                torch.abs(attr)
            )  # The absolute value of the GradShap attribution gives the feature importance
        return attr


class SVS:
    def __init__(self, f):
        self.f = f

    def attribute(self, X, normalize=True):
        explainer = ShapleyValueSampling(forward_func=self.f)
        baseline = torch.mean(X, dim=0, keepdim=True)
        attr = explainer.attribute(X, baselines=baseline)
        if normalize:
            attr = normal(torch.abs(attr))  # The absolute value of the SVS attribution gives the feature importance
        return attr