class InceptionDiscriminator(nn.Module):
    def __init__(self, input_nc, use_sigmoid=False, gpu_ids=[]):
        super(InceptionDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids

        self.patchD = PatchD(input_nc=input_nc)
        self.inception1 = InceptionA(256)
        self.inception2 = InceptionB(384)
        self.inception3 = InceptionB(192)
        self.last2 = nn.Conv2d(192, 1, kernel_size=3, padding=1)
        self.last1 = nn.Sigmoid()

    def forward(self, x):
        if len(self.gpu_ids) and isinstance(x.data, torch.cuda.FloatTensor):
            x = nn.parallel.data_parallel(self.patchD, x, self.gpu_ids)
            x = nn.parallel.data_parallel(self.inception1, x, self.gpu_ids)
            x = nn.parallel.data_parallel(self.inception2, x, self.gpu_ids)
            x = nn.parallel.data_parallel(self.inception3, x, self.gpu_ids)
            x = nn.parallel.data_parallel(self.last2, x, self.gpu_ids)
            return nn.parallel.data_parallel(self.last1, x, self.gpu_ids)
        else:
            out = self.patchD(x)
            out = self.inception1(out)
            out = self.inception2(out)
            out = self.inception3(out)
            out = self.last2(out)
            out = self.last1(out)
            return out

class PatchD(nn.Module):
    def __init__(self, input_nc, norm_layer=nn.BatchNorm2d):
        super(PatchD, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.sequence = [
            nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(256),
            nn.LeakyReLU(0.2, True)
        ]
        self.model = nn.Sequential(*self.sequence)

    def forward(self, x):
        return self.model(x)

class InceptionA(nn.Module):
    def __init__(self, input_nc):
        super(InceptionA, self).__init__()
        self.branch1x1 = nn.Conv2d(input_nc, 128, kernel_size=1)

        self.branch3x3_1 = nn.Conv2d(input_nc, 64, kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.branch5x5_1 = nn.Conv2d(input_nc, 64, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.branch5x5_3 = nn.Conv2d(128, 128, kernel_size=3 ,padding=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch5x5 = self.branch5x5_3(branch5x5)

        out = [branch1x1, branch3x3, branch5x5]
        return torch.cat(out, 1)

class InceptionB(nn.Module):
    def __init__(self, input_nc):
        super(InceptionB, self).__init__()
        self.branch1x1 = nn.Conv2d(input_nc, 64, kernel_size=1)

        self.branch3x3_1 = nn.Conv2d(input_nc, 64, kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.branch5x5_1 = nn.Conv2d(input_nc, 64, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.branch5x5_3 = nn.Conv2d(64, 64, kernel_size=3 ,padding=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch5x5 = self.branch5x5_3(branch5x5)

        out = [branch1x1, branch3x3, branch5x5]
        return torch.cat(out, 1)
