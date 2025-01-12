from torch.optim import Optimizer
import copy


class SVRG(Optimizer):
    """
        This class is for calculating the gradient of one iteration.
        
        - params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        - lr (float): learning rate
    """
    def __init__(self, params, lr):
        print("Using SVRG optimizer ...")
        self.mu = None 
        defaults = dict(lr=lr)
        super(SVRG, self).__init__(params, defaults)
    
    def get_param_groups(self):
            return self.param_groups

    def set_mu(self, new_mu):
        """
            Set the mean gradient for the current iteration. 
        """
        if self.mu is None:
            self.mu = copy.deepcopy(new_mu)
        for u_group, new_group in zip(self.mu, new_mu):  
            for mu, new_mu in zip(u_group['params'], new_group['params']):
                mu.grad = new_mu.grad.clone()

    def step(self, params):
        """
            Performs a single optimization step via SVRG.
            (Hint: This part is similar to that in SGD)
        """
        for group, new_group, u_group in zip (self.param_groups, params, self.mu):
            lr = group['lr']
            for p, q, u in zip(group['params'], new_group['params'], u_group['params']):
                if p.grad is None or q.grad is None or u.grad is None:
                    continue
                variance_reduced_grad = p.grad.data - q.grad.data + u.grad.data
                p.data -= lr * variance_reduced_grad
        
class SVRG_Snapshot(Optimizer):
    """
        This class for calculating the average gradient (i.e., snapshot) of all samples.

        - params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        - lr (float): learning rate
    """
    def __init__(self, params):
        defaults = dict()
        super(SVRG_Snapshot, self).__init__(params, defaults)
      
    def get_param_groups(self):
            return self.param_groups
    
    def set_param_groups(self, new_params):
        """
            Copies the parameters from the inner-loop optimizer. 
            There are two options:
            1. Use the latest parameters
            2. Draw uniformly at random from the previous m parameters
        """
        for group, new_group in zip(self.param_groups, new_params): 
            for p, q in zip(group['params'], new_group['params']):
                  p.data[:] = q.data[:]