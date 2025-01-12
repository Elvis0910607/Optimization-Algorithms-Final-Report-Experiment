import torch
from torch.optim import Optimizer
import copy

class STORC(Optimizer):
    """
    STOchastic variance-Reduced Conditional gradient sliding (STORC) optimizer.
    """

    def __init__(self, params, lr, beta):
        """
        Initialize STORC optimizer.

        Args:
            params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
            lr (float): Learning rate.
            beta (float): Beta parameter for STORC algorithm, controlling the variance reduction.
        """
        print("Using STORC optimizer ...")
        self.beta = beta  
        self.mu = None  
        self.snapshot_params = None  
        self.z = None   

        defaults = dict(lr=lr)
        super(STORC, self).__init__(params, defaults)

    def get_param_groups(self):
        """
        Return the parameter groups.
        """
        return self.param_groups

    def set_snapshot(self, params):
        """
        Sets the snapshot parameters and calculates their gradients.
        """
        self.snapshot_params = copy.deepcopy(params)
        self.mu = copy.deepcopy(params)

    def step(self):
        """
        Perform a single optimization step via STORC.
        """
        for group in self.param_groups:
            lr = group['lr']

            if self.z is None:
                self.z = [torch.zeros_like(p.data) for p in group['params']]

            for idx, p in enumerate(group['params']):
                if p.grad is None:
                    continue

                grad = p.grad.data
                variance_reduced_grad = grad * self.beta

                gamma = 2 / (1 + (1 + 4 / (self.beta * (1 / lr)))**0.5)
                self.z[idx].data = (1 - gamma) * self.z[idx].data + gamma * p.data

                p.data.add_(-lr, variance_reduced_grad)
                p.data.copy_(self.z[idx].data - lr * variance_reduced_grad)

