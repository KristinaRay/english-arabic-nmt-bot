import torch
import torch.nn as nn

from config import *


class HardConcreteGate(nn.Module):
    """
    A gate made of stretched concrete distribution 
    reference:
    LEARNING SPARSE NEURAL NETWORKS THROUGH L0 REGULARIZATION 
    https://openreview.net/pdf?id=H1Y8hhg0b
    """

    def __init__(self,
                 num_heads,
                 temperature=GATE_TEMPERATURE ,
                 stretch_interval=(-0.1, 1.1),
                 l0_penalty=GATE_L0_PENALTY,
                 l2_penalty=0.0,
                 eps=1e-9,
                 hard=False):
        super().__init__()
        
        self.num_heads = num_heads
        self.log_a = nn.Parameter(torch.full((1, num_heads, 1, 1), 0.0)) 
        self.temperature = temperature
        self.stretch_interval = stretch_interval
        self.l0_penalty = l0_penalty
        self.l2_penalty = l2_penalty
        self.eps = eps
        self.hard = hard                      
        self.sigmoid = nn.Sigmoid()
                                  
    def forward(self, values, is_train):
        """
        Applies gate to values, if is_train
        """
        self.gates = self.get_gates(values, is_train)
                                  
        return values * self.gates.to(device)
                                  
    def penalty(self):
        """
        Computes l0 and l2 penalties. For l2 penalty one must also provide the sparsified values
        (usually activations or weights) before they are multiplied by the gate
        Returns the regularizer value that should to be MINIMIZED (negative logprior)
        """
        assert self.l0_penalty == 0 or self.l2_penalty == 0,  "get_penalty() is called with both penalties set to 0"
        low, high = self.stretch_interval
        assert low < 0.0, "can be computed only if lower stretch limit is negative"
                                  
        hard_sigmoid = self.sigmoid(self.log_a - self.temperature * torch.log(torch.tensor(-low / high)))
        gate_open = torch.clamp(hard_sigmoid, min=self.eps, max=1-self.eps)
        
        return gate_open 
                                  
                                  
    def get_gates(self, values, is_train):
        """ 
        samples gate activations in [0, 1] interval
        
        """
        low, high = self.stretch_interval
        if is_train:
            shape = self.log_a.shape if values==None else values.shape
            noise = ((1.0 - self.eps) * torch.rand(shape) + self.eps).to(device)
            concrete = self.sigmoid((torch.log(noise) - torch.log(1 - noise) + self.log_a) / self.temperature)
        else:
            concrete = self.sigmoid(self.log_a)

        stretched_concrete = concrete * (high - low) + low
        clipped_concrete = torch.clamp(stretched_concrete,  min=0, max=1)
        if self.hard:
            hard_concrete = (torch.greater(clipped_concrete, 0.5)).float()
            clipped_concrete = clipped_concrete + (hard_concrete - clipped_concrete).detach()
                                  
        return clipped_concrete
