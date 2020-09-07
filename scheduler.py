import torch
import numpy as np

class Lambda_LR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)

#a replay buffer to augment generated data
class Buffer:
    def __init__(self, device='cpu', max_size=50):
        assert max_size > 0
        self.device = device
        self.max_size = max_size
        self.data = []

    def augment(self, data):
        output = []
        for component in data.data:
            component = torch.unsqueeze(component, 0)
            if len(self.data) < self.max_size:
                self.data.append(component)
                output.append(component)
            else:
                if np.random.uniform(0, 1) > 0.5:
                    i = np.random.randint(0, self.max_size - 1)
                    output.append(self.data[i].clone())
                    self.data[i] = component
                else:
                    output.append(component)
        return torch.autograd.Variable(torch.cat(output)).to(self.device)