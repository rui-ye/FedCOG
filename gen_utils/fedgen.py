import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from kornia import augmentation
from torchvision import transforms
from tqdm import tqdm
import torchvision.utils as vutils

def kldiv(logits, targets, T=1.0, reduction='batchmean'):
    q = F.log_softmax(logits / T, dim=1)
    p = F.softmax(targets / T, dim=1)
    return F.kl_div(q, p, reduction=reduction) * (T * T)

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



class GenImageDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __getitem__(self, idx):
        img = self.data[idx]
        if self.transform:
            img = self.transform(img)
        return img

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
        self.adv = adv
        self.bn = bn
        self.oh = oh
        self.num_classes = num_classes
        self.synthesis_batch_size = synthesis_batch_size
        self.sample_batch_size = sample_batch_size
        self.data_iter = None
        self.teacher = teacher
        self.dataset = dataset

        self.generator = generator.cuda().train()

        self.aug = MultiTransform([
            # global view
            transforms.Compose([
                augmentation.RandomCrop(size=[self.img_size, self.img_size], padding=4),
                augmentation.RandomHorizontalFlip(),
            ]),
            # local view
            transforms.Compose([
                augmentation.RandomResizedCrop(size=[self.img_size, self.img_size], scale=[0.25, 1.0]),
                augmentation.RandomHorizontalFlip(),
            ]),
        ])
        # =======================
        if not ("cifar" in dataset):
            self.transform = transforms.Compose(
                [   transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ])
        else:
            self.transform = transforms.Compose(
                [   transforms.ToPILImage(),
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ])


    def synthesize(self):
        net = self.teacher
        best_cost = 1e6
        best_inputs = None
        targets = []
        for i in range(self.num_classes):
            targets.append(i * torch.ones(int(self.synthesis_batch_size * self.distribution[i])).long())
        targets = torch.cat(targets, dim = 0)
        # targets = torch.randint(low=0, high=self.num_classes, size=(self.synthesis_batch_size,))
        # targets = targets.sort()[0]
        targets = targets.cuda()
        reset_model(self.generator)
        z = torch.randn(size=(targets.shape[0], self.nz)).cuda()  #
        z.requires_grad = True
        optimizer = torch.optim.Adam([{'params': self.generator.parameters()}, {'params': [z]}], self.lr_g,
                                     betas=[0.5, 0.999])
        hooks = []
        net.eval()
        for m in net.modules():
            if isinstance(m, nn.BatchNorm2d):
                hooks.append(DeepInversionHook(m))

        for it in range(self.iterations):
            optimizer.zero_grad()

            inputs = self.generator(z)  # bs,nz
            global_view, _ = self.aug(inputs)  # crop and normalize
            #############################################
            # Gate
            net.cuda()
            t_out = net(global_view)

            loss_bn = sum([h.r_feature for h in hooks])  # bn层loss
            loss_oh = F.cross_entropy(t_out, targets)  # ce_loss
            
            self.student.eval()
            s_out = self.student(global_view)
            mask = (s_out.max(1)[1] != t_out.max(1)[1]).float()
            loss_adv = -(kldiv(s_out, t_out, reduction='none').sum(
                1) * mask).mean()  # decision adversarial distillation

            loss = self.bn * loss_bn + self.oh * loss_oh + self.adv * loss_adv

            if best_cost > loss.item() or best_inputs is None:
                best_cost = loss.item()
                best_inputs = inputs.data

            loss.backward()
            optimizer.step()
            # optimizer_mlp.step()
        print('loss_ce:{}, loss_div:{}'.format(loss_oh.item(), loss_adv.item()))
        # vutils.save_image(best_inputs.clone(), '1.png', normalize=True, scale_each=True, nrow=10)

        # save best inputs and reset data iter
        datasets = GenImageDataset(best_inputs.cpu(), self.transform)
        data_loader = DataLoader(
            datasets, batch_size=self.sample_batch_size, shuffle=True,
            num_workers=0, pin_memory=True, )
        return data_loader


