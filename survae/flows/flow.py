import torch
from torch import nn
from collections.abc import Iterable
from survae.distributions import Distribution
from survae.transforms import Transform

REPORT = False
def info(module, enabled=False):

    def print_info(x):
        
        print(type(module))
        print(x.shape)
        x = result = module(x)
        if isinstance(result, tuple):
            x = result[0]

        print(x.shape)

        return result
    if enabled:
        return print_info
    return module

class Flow(Distribution):
    """
    Base class for Flow.
    Flows use the forward transforms to transform data to noise.
    The inverse transforms can subsequently be used for sampling.
    These are typically useful as generative models of data.
    """

    def __init__(self, base_dist, transforms):
        super(Flow, self).__init__()
        assert isinstance(base_dist, Distribution)
        if isinstance(transforms, Transform): transforms = [transforms]
        assert isinstance(transforms, Iterable)
        assert all(isinstance(transform, Transform) for transform in transforms)
        self.base_dist = base_dist
        self.transforms = nn.ModuleList(transforms)
        self.lower_bound = any(transform.lower_bound for transform in transforms)

    def log_prob(self, x):
        log_prob = torch.zeros(x.shape[0], device=x.device)
        for transform in self.transforms:
            x, ldj = info(transform, REPORT)(x)
            log_prob += ldj
        log_prob += self.base_dist.log_prob(x)
        return log_prob

    def sample(self, num_samples):
        z = self.base_dist.sample(num_samples)
        for transform in reversed(self.transforms):
            z = info(transform.inverse, REPORT)(z)

        return z

    def sample_with_log_prob(self, num_samples):
        raise RuntimeError("Flow does not support sample_with_log_prob, see InverseFlow instead.")
