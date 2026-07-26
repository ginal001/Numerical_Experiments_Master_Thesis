import math
import torch


def sample_ring_pt(n_samples, r_min, r_max):
    theta = torch.rand(n_samples) * 2 * math.pi
    
    u = torch.rand(n_samples)
    r = torch.sqrt(u * (r_max**2 - r_min**2) + r_min**2)
    
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    
    return torch.stack([x, y], dim=1)

def get_gradient_field(phi_net, x, t):
    with torch.set_grad_enabled(True):
        x = x.detach().requires_grad_(True)
        phi = phi_net(x, t)
        grad_phi = torch.autograd.grad(
            outputs=phi, inputs=x, grad_outputs=torch.ones_like(phi), create_graph=True
        )[0]
    return grad_phi

def solve_ode(x_start, velocity, steps=100):
    x = x_start.clone()
    n_samples = x.shape[0]
    dt = 1.0 / steps
    
    for step in range(steps):
        t_val = step * dt
        t_tensor = torch.full((n_samples, 1), fill_value=t_val, dtype=torch.float32, device=x.device)
        
        v = velocity(x, t_tensor)
        
        x = x + v * dt
        
    return x
