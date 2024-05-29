import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, img_size=32, nc=3):
        super(Generator, self).__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(nz, ngf * 2 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(ngf * 2),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf * 2, ngf * 2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf * 2, ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img
    
class AdvSynthesizer():
    def __init__(self, teacher, student, generator, nz, num_classes, img_size,
                 iterations, lr_g,
                 synthesis_batch_size, sample_batch_size,
                 adv, bn, oh, dataset, distribution):
        super(AdvSynthesizer, self).__init__()
        self.distribution = distribution
        self.student = student
        self.img_size = img_size
        self.iterations = iterations
        self.lr_g = lr_g
        self.nz = nz
        
class DeepInversionHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = module

    def hook_fn(self, module, input, output):  # hook_fn(module, input, output) -> None
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)
        # forcing mean and variance to match between two distributions
        # other ways might work better, i.g. KL divergence
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
            module.running_mean.data - mean, 2)
        self.r_feature = r_feature

    def remove(self):
        self.hook.remove()

class GenImageDataset(Dataset):
    def __init__(self, data, output, transform=None):
        self.data = data
        self.output = output
        self.transform = transform

    def __getitem__(self, idx):
        img = self.data[idx]
        if self.transform:
            img = self.transform(img)
        label = self.output[idx]
        return img, label

    def __len__(self):
        return self.data.shape[0]
    
class MultiTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [t(x) for t in self.transform]

    def __repr__(self):
        return str(self.transform)
    
def reset_model(model):
    for m in model.modules():
        if isinstance(m, (nn.ConvTranspose2d, nn.Linear, nn.Conv2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, (nn.BatchNorm2d)):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)


def kldiv(logits, targets, T=1.0, reduction='batchmean'):
    q = F.log_softmax(logits / T, dim=1)
    p = F.softmax(targets / T, dim=1)
    return F.kl_div(q, p, reduction=reduction) * (T * T)

def jensen_shanon_div(s_out, t_out, T = 1.0, eps = 0.0):
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    # Jensen Shanon divergence:
    # another way to force KL between negative probabilities
    P = nn.functional.softmax(s_out / T, dim=1)
    Q = nn.functional.softmax(t_out / T, dim=1)
    M = 0.5 * (P + Q)

    P = torch.clamp(P, 0.01, 0.99)
    Q = torch.clamp(Q, 0.01, 0.99)
    M = torch.clamp(M, 0.01, 0.99)
    loss_adv = 0.5 * kl_loss(torch.log(P + eps), M) + 0.5 * kl_loss(torch.log(Q + eps), M)
        # JS criteria - 0 means full correlation, 1 - means completely different
    loss_adv = 1.0 - torch.clamp(loss_adv, 0.0, 1.0)
    return loss_adv

class KLDiv(nn.Module):
    def __init__(self, T=1.0, reduction='batchmean'):
        """

        :rtype: object
        """
        super().__init__()
        self.T = T
        self.reduction = reduction

    def forward(self, logits, targets):
        return kldiv(logits, targets, T=self.T, reduction=self.reduction)

