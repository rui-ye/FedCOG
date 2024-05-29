import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from kornia import augmentation
from torchvision import transforms
from gen_utils.generate_utils import DeepInversionHook, GenImageDataset, MultiTransform, reset_model, jensen_shanon_div, Generator
import torchvision.utils as vutils
import time

class ImageSynthesizer():
    def __init__(self, args, teacher, student, nz, num_classes, img_size,
                 iterations, lr_g,
                 synthesis_batch_size, sample_batch_size,
                 adv, bn, oh, dataset, distribution, hard_label=False):
        super(ImageSynthesizer, self).__init__()
        self.args = args
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
        self.hard_label = hard_label

        if self.args.generator:
            self.generator = Generator(nz=args.nz, ngf=64, img_size=args.image_size, nc=args.channel).cuda()


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

        if self.args.gen_aug:
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
        else:
            self.transform = None

    def synthesize(self):
        net = self.teacher
        best_cost = 1e6
        best_inputs = None
        targets = []
        if self.args.gen_complement:
            tensor1 = torch.arange(0, 10)
            tensor2 = torch.arange(0, 10)

            # concatenate the tensors
            targets = torch.cat((tensor1, tensor2), dim=0)
        else:
            for i in range(self.num_classes):
                targets.append(i * torch.ones(int(self.synthesis_batch_size * self.distribution[i])).long())
            targets = torch.cat(targets, dim = 0)

        targets = targets.cuda()
        if self.args.generator:
            reset_model(self.generator)
            z = torch.randn(size=(targets.shape[0], self.nz)).cuda()  #
            z.requires_grad = True
            optimizer = torch.optim.Adam([{'params': self.generator.parameters()}, {'params': [z]}], self.lr_g,
                                        betas=[0.5, 0.999])
        else:
            inputs = torch.randn((targets.shape[0], 3, self.img_size, self.img_size), requires_grad=True, device="cuda")
            optimizer = torch.optim.Adam([inputs], lr=self.lr_g, betas=[0.5, 0.9], eps = 1e-8)
        if self.args.gen_downsample:
            pooling_function = nn.modules.pooling.AvgPool2d(kernel_size=2)
        
        hooks = []
        net.eval()
        for m in net.modules():
            if isinstance(m, nn.BatchNorm2d):
                hooks.append(DeepInversionHook(m))

        start_time = time.time()
        for it in range(self.iterations):
            optimizer.zero_grad()
            if self.args.generator:
                inputs = self.generator(z)  # bs,nz
            if self.args.gen_aug:
                inputs_aug, _ = self.aug(inputs)  # crop and normalize
            else:
                inputs_aug = inputs #rename to keep pointer of  inputs
            #for resnet with adaptive pooling
            if self.args.gen_downsample and it < self.iterations / 2:
                inputs_aug = pooling_function(inputs_aug)
            #############################################
            # Gate
            net.cuda()
            t_out = net(inputs_aug)

            loss_bn = sum([h.r_feature for h in hooks])  # bn层loss
            loss_oh = F.cross_entropy(t_out, targets)  # ce_loss
            
            self.student.eval()
            s_out = self.student(inputs_aug)
            
            loss_adv = jensen_shanon_div(s_out, t_out, self.args.js_T)

            loss = self.bn * loss_bn + self.oh * loss_oh + self.adv * loss_adv

            if best_cost > loss.item() or best_inputs is None:
                best_cost = loss.item()
                best_inputs = inputs.data
                best_out = t_out.data

            loss.backward()
            optimizer.step()
        try:
            print('loss_ce:{}, loss_div:{}, loss_bn:{}'.format(loss_oh.item(), loss_adv.item(), loss_bn))
        except:
            best_inputs = inputs.data
            net.cuda()
            best_out = net(inputs).data

        # save best inputs and reset data iter
        if self.hard_label:
            datasets = GenImageDataset(best_inputs.cpu(), best_out.argmax(dim=1).cpu(), self.transform)
        else:
            datasets = GenImageDataset(best_inputs.cpu(), best_out.cpu(), self.transform)
        data_loader = DataLoader(
            datasets, batch_size=self.sample_batch_size, shuffle=True,
            num_workers=4, pin_memory=True, )
        return data_loader
    

