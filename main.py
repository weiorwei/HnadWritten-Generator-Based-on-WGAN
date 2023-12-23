import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch import optim
from torch.utils.data import DataLoader
from own_layer import *
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

Lambda = 10
batch_size = 80
num_dim = 80
epoch = 300

train_data = torchvision.datasets.MNIST(root='../data', train=True, download=True,
                                        transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.MNIST(root='../data', train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

train_size = len(train_data)
test_size = len(test_data)
train = int(train_size * 0.022)
test = int(train_size * 0.978)
train_data, test_data = torch.utils.data.random_split(train_data, [train, test])

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, sampler=True)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        momentum = 0.9
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(5, 5), stride=(2, 2), padding=16),
            own_relu(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(2, 2), padding=16),
            own_relu(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5, 5), stride=(2, 2), padding=16),
            own_relu(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(5, 5), stride=(2, 2), padding=16),
            own_relu(),
            nn.Flatten(),
            nn.Linear(401408, 1)
        )

    def forward(self, x):
        x = self.model(x)
        return x


class Generator(nn.Module):
    def __init__(self):
        d = 3
        momentum = 0.9
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(batch_size, d * d * 512),
            Reshape((batch_size, 512, d, d)),
            nn.BatchNorm2d(512, momentum=momentum),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, (5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1)),
            nn.BatchNorm2d(256, momentum=momentum),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, (5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1)),
            nn.BatchNorm2d(128, momentum=momentum),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, (5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1)),
            nn.BatchNorm2d(64, momentum=momentum),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, (5, 5)),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.model(x)
        return x


def gradient_penalty(model, real, fake):
    t = torch.rand(batch_size,1,1,1).cuda()
    t = t.expand_as(real)
    mid = t * real + (1 - t) * fake
    mid = mid.requires_grad_()
    pred_mid = model(mid)
    grads = torch.autograd.grad(outputs=pred_mid, inputs=mid,
                                grad_outputs=torch.ones_like(pred_mid),
                                create_graph=True, retain_graph=True,
                                only_inputs=True)[0]
    gp = torch.pow(torch.norm(grads).sqrt() - 1, 2).mean()
    return gp


discriminate = Discriminator().cuda()
generate = Generator().cuda()
writer = SummaryWriter('logs')

learn_rate = 0.0002
loss_fn = nn.BCELoss().cuda()
optimizerD = optim.Adam(discriminate.parameters(), lr=learn_rate, betas=(0.5, 0.999))
optimizerG = optim.Adam(generate.parameters(), lr=learn_rate, betas=(0.5, 0.999))
flag = 0
for i in range(1, epoch):
    for img, target in train_dataloader:
        noise = torch.randn((batch_size, num_dim), device='cuda')
        img = img.cuda()
        genrate_img = generate(noise)
        # img_show_1=genrate_img[0].cpu().detach()
        # img_show=np.transpose(img_show_1,(1,2,0))
        # plt.imshow(img_show,cmap='gray')
        # plt.show()

        optimizerD.zero_grad()
        out_real = discriminate(img)
        out_fake = discriminate(genrate_img).detach()
        Discriminator_loss_input_real = -torch.mean(out_real)
        Discriminator_loss_input_fake = torch.mean(out_fake)
        gp = gradient_penalty(discriminate, img, genrate_img)
        Discriminator_loss_total = Discriminator_loss_input_real + Discriminator_loss_input_fake + Lambda * gp
        Discriminator_loss_total.backward()
        optimizerD.step()

        if i % 3 == 0:
            optimizerG.zero_grad()
            out_fake_g = discriminate(genrate_img)
            Generator_loss = -torch.mean(out_fake_g)
            Generator_loss.backward()
            optimizerG.step()

    if i % 2 == 0:
        print("第{}轮训练".format(i))
        print('Discriminator_loss_total:{}'.format(Discriminator_loss_total.item()))
        print('Generator_loss:{}'.format(Generator_loss.item()))
        writer.add_scalars('loss_total', {'Discriminator_loss_total': Discriminator_loss_total.item(),
                                          'Generator_loss': Generator_loss}, i)
        torch.save(discriminate, 'discriminate{}_loss{}.pth'.format(i, Discriminator_loss_total))
        torch.save(generate, 'generate{}_loss{}.pth'.format(i, Generator_loss))
        print("第{}轮训练模型已保存".format(i))
writer.close()
