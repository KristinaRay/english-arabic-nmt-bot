import torch
import torch.nn as nn


class HardConcreteGate(nn.Module):
    """
    A gate made of stretched concrete distribution 
    reference:
    LEARNING SPARSE NEURAL NETWORKS THROUGH L0 REGULARIZATION https://openreview.net/pdf?id=H1Y8hhg0b
    """

    def __init__(self,
                 num_heads,
                 log_a=0.0,
                 temperature=0.5,
                 stretch_interval=(-0.1, 1.1),
                 l0_penalty_lambda=0.0,
                 l2_penalty_lambda=0.0,
                 eps=1e-9,
                 hard=False):
        super().__init__()
        
        self.num_heads = num_heads
        self.log_a = nn.Parameter(torch.full((num_heads,0), log_a) # location of the head
        self.temperature = temperature
        self.stretch_interval = stretch_interval
        self.l0_penalty = l0_penalty
        self.l2_penalty = l2_penalty
        self.eps = eps
        self.hard = hard                      
        self.sigmoid = nn.Sigmoid()
                                  
    def forward(self, values, is_train=None):
        """
        Applies gate to values, if is_train
        """
        
        gates = self.get_gates(is_train, shape=self.log_a.shape)

        if self.l0_penalty != 0 or self.l2_penalty != 0:
            reg = self.get_penalty(values=values)
                                  
        return values * gates, reg
                                  
    def get_penalty(self, values=None, axis=None):
        """
        Computes l0 and l2 penalties. For l2 penalty one must also provide the sparsified values
        (usually activations or weights) before they are multiplied by the gate
        Returns the regularizer value that should to be MINIMIZED (negative logprior)
        """
        assert self.l0_penalty == self.l2_penalty == 0,  "get_penalty() is called with both penalties set to 0"
        low, high = self.stretch_interval
        assert low < 0.0, "can be computed only if lower stretch limit is negative"
                                  
        hard_sigmoid = self.sigmoid(self.log_a - self.temperature * torch.log(-low / high))
        gate_open = torch.clamp(hard_sigmoid, min=self.eps, max=1-self.eps)
        # print('gate.open.shape', gate.open.shape)                      
        total_reg = 0.0
        if self.l0_penalty != 0:
            if values != None:
                gate_open = torch.add(gate_open, torch.zeros_like(values)) # broadcast shape to account for values
                l0_reg = self.l0_penalty * gate_open.sum()
                
                total_reg += torch.mean(l0_reg, dim=0)
        if self.l2_penalty != 0:
            if values != None:
                l2_reg = 0.5 * self.l2_penalty * gate_open * (values ** 2).sum()
                total_reg += torch.mean(l2_reg, dim=0)

        return total_reg                         
         
                                  
                                  
    def get_gates(self, is_train, shape=None):
        """ 
        samples gate activations in [0, 1] interval
        
        """
        low, high = self.stretch_interval
        if is_train:
            shape = self.log_a.shape if shape is None else shape
            noise = (1.0 - self.eps) * torch.rand(shape) + self.eps
            concrete = self.sigmoid((torch.log(noise) - torch.log(1 - noise) + self.log_a) / self.temperature)
        else:
            concrete = self.sigmoid(self.log_a)

        stretched_concrete = concrete * (high - low) + low
        clipped_concrete = torch.clamp(stretched_concrete,  min=0, max=1)
        if self.hard:
            hard_concrete = (torch.greater(clipped_concrete, 0.5)).float()
            clipped_concrete = clipped_concrete + (hard_concrete - clipped_concrete).detach()
                                  
        return clipped_concrete

          