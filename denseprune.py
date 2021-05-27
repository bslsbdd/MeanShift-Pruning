import os
import argparse
import numpy as np
import torch
import math
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from sklearn.cluster import MeanShift, estimate_bandwidth
import time
from thop import profile
import torch.nn.functional as F
#from torchsummary import summary
class Transition(nn.Module):
    def __init__(self, inplanes, outplanes, cfg):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.select = channel_selection(inplanes)
        self.conv1 = nn.Conv2d(cfg, outplanes, kernel_size=1,
                               bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.bn1(x)
        out = self.select(out)
        out = self.relu(out)
        out = self.conv1(out)
        out = F.avg_pool2d(out, 2)
        return out

class channel_selection(nn.Module):
    def __init__(self, num_channels):
        super(channel_selection, self).__init__()
        self.indexes = nn.Parameter(torch.ones(num_channels))

    def forward(self, input_tensor):
        selected_index = np.squeeze(np.argwhere(self.indexes.data.cpu().numpy()))
        if selected_index.size == 1:
            selected_index = np.resize(selected_index, (1,)) 
        output = input_tensor[:, selected_index, :, :]
        return output
    
class BasicBlock(nn.Module):
    def __init__(self, inplanes, cfg, expansion=1, growthRate=12, dropRate=0):
        super(BasicBlock, self).__init__()
        planes = expansion * growthRate
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.select = channel_selection(inplanes)
        self.conv1 = nn.Conv2d(cfg, growthRate, kernel_size=3, 
                               padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropRate = dropRate

    def forward(self, x):
        out = self.bn1(x)
        out = self.select(out)
        out = self.relu(out)
        out = self.conv1(out)
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)

        out = torch.cat((x, out), 1)

        return out
    
class densenet(nn.Module):

    def __init__(self, depth=40, 
        dropRate=0, dataset='cifar10', growthRate=12, compressionRate=1, cfg = None):
        super(densenet, self).__init__()

        assert (depth - 4) % 3 == 0, 'depth should be 3n+4'
        n = (depth - 4) // 3
        block = BasicBlock

        self.growthRate = growthRate
        self.dropRate = dropRate

        if cfg == None:
            cfg = []
            start = growthRate*2
            for _ in range(3):
                cfg.append([start + growthRate*i for i in range(n+1)])
                start += growthRate*n
            cfg = [item for sub_list in cfg for item in sub_list]

        assert len(cfg) == 3*n+3, 'length of config variable cfg should be 3n+3'

        self.inplanes = growthRate * 2
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1,
                               bias=False)
        self.dense1 = self._make_denseblock(block, n, cfg[0:n])
        self.trans1 = self._make_transition(compressionRate, cfg[n])
        self.dense2 = self._make_denseblock(block, n, cfg[n+1:2*n+1])
        self.trans2 = self._make_transition(compressionRate, cfg[2*n+1])
        self.dense3 = self._make_denseblock(block, n, cfg[2*n+2:3*n+2])
        self.bn = nn.BatchNorm2d(self.inplanes)
        self.select = channel_selection(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)

        if dataset == 'cifar10':
            self.fc = nn.Linear(cfg[-1], 10)
        elif dataset == 'cifar100':
            self.fc = nn.Linear(cfg[-1], 100)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()

    def _make_denseblock(self, block, blocks, cfg):
        layers = []
        assert blocks == len(cfg), 'Length of the cfg parameter is not right.'
        for i in range(blocks):
            layers.append(block(self.inplanes, cfg = cfg[i], growthRate=self.growthRate, dropRate=self.dropRate))
            self.inplanes += self.growthRate

        return nn.Sequential(*layers)

    def _make_transition(self, compressionRate, cfg):
        inplanes = self.inplanes
        outplanes = int(math.floor(self.inplanes // compressionRate))
        self.inplanes = outplanes
        return Transition(inplanes, outplanes, cfg)

    def forward(self, x):
        x = self.conv1(x)

        x = self.trans1(self.dense1(x))
        x = self.trans2(self.dense2(x))
        x = self.dense3(x)
        x = self.bn(x)
        x = self.select(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def test(model):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'cifar10':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data.cifar10', train=False, download= True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)
    elif args.dataset == 'cifar100':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)
    else:
        raise ValueError("No valid dataset is given.")
    model.eval()
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(
        correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))
# Prune settings
parser = argparse.ArgumentParser(description='PyTorch Meanshift CIFAR prune')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--depth', type=int, default=40,
                    help='depth of the resnet')
parser.add_argument('--model', default='densent_model_best.pth.tar', type=str, metavar='PATH',
                    help='path to the model (default: none)')
parser.add_argument('--save', default='weights/', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if not os.path.exists(args.save):
    os.makedirs(args.save)

model = densenet(depth=args.depth, dataset=args.dataset)

if args.cuda:
    model.cuda()
if args.model:
    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        #checkpoint = torch.load(args.model,map_location=torch.device('cpu')
        checkpoint = torch.load(args.model,map_location=torch.device('cpu'))
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.model, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))



data0 = torch.randn(1*3,32*32)
data0 = data0.reshape(1,3,32,32)
a0=time.time()
predict0=model(data0)
a1=time.time()
time_old = a1-a0
input = torch.randn(1, 3, 32, 32)
flop, para = profile(model,inputs=(input, ))
print("%.2fM" % (flop/1e6), "%.2fM" % (para/1e6))

thre_all = []
kk = []
for k,m in enumerate(model.modules()):
    if isinstance(m,nn.BatchNorm2d):
        size = m.weight.data.shape[0]
        bn = torch.zeros(size)
        bn = m.weight.data.abs().clone()
        bn = bn.reshape(-1,1).float().cpu()
        bandwidth = estimate_bandwidth(bn, quantile=0.3, n_samples=len(bn))
        meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        meanshift.fit(bn)
        centroids = meanshift.cluster_centers_
        labels = meanshift.labels_
        cluster_num = len(np.unique(labels))
        thre_min = np.min(centroids)
        thre_max = np.max(centroids)
        thre_mean = np.mean(centroids)
        thre_median = np.median(centroids)
        print(centroids)
        print('cluster num:{}'.format(cluster_num))
        print('min:{}'.format(thre_min))
        print('max:{}'.format(thre_max))
        print('mean:{}'.format(thre_mean))
        print('median:{}'.format(thre_median))
        thre = thre_min       
        thre_all.append(thre)
        kk.append(k)
        
pruned = 0
size_all = 0
cfg = []
cfg_mask = []
n = 0
for k, m in enumerate(model.modules()):
    if isinstance(m, nn.BatchNorm2d):
        if n == 0:thre = 0
        else: thre = thre_all[n]
        size = m.weight.data.shape[0]
        weight_copy = m.weight.data.abs().clone()
        mask = weight_copy.gt(thre).float()
        pruned = pruned + mask.shape[0] - torch.sum(mask)
        m.weight.data.mul_(mask)
        m.bias.data.mul_(mask)
        cfg.append(int(torch.sum(mask)))
        cfg_mask.append(mask.clone())
        n += 1
        size_all += size
        print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
            format(k, mask.shape[0], int(torch.sum(mask))))
    elif isinstance(m, nn.MaxPool2d):
        cfg.append('M')

pruned_ratio = pruned/size_all
print(pruned_ratio)
print('Pre-prune Successful!')


print("Cfg:")
print(cfg)

newmodel = densenet(depth=args.depth, dataset=args.dataset, cfg=cfg)

if args.cuda:
    newmodel.cuda()

num_parameters = sum([param.nelement() for param in newmodel.parameters()])
savepath = os.path.join(args.save, "prune.txt")
with open(savepath, "w") as fp:
    fp.write("Configuration: \n"+str(cfg)+"\n")
    fp.write("Number of parameters: \n"+str(num_parameters)+"\n")
    #fp.write("Test accuracy: \n"+str(acc))

old_modules = list(model.modules())
new_modules = list(newmodel.modules())

layer_id_in_cfg = 0
start_mask = torch.ones(3)
end_mask = cfg_mask[layer_id_in_cfg]
first_conv = True

for layer_id in range(len(old_modules)):
    m0 = old_modules[layer_id]
    m1 = new_modules[layer_id]
    if isinstance(m0, nn.BatchNorm2d):
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        if idx1.size == 1:
            idx1 = np.resize(idx1,(1,))

        if isinstance(old_modules[layer_id + 1], channel_selection):
            # If the next layer is the channel selection layer, then the current batch normalization layer won't be pruned.
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()
            m1.running_mean = m0.running_mean.clone()
            m1.running_var = m0.running_var.clone()

            # We need to set the mask parameter `indexes` for the channel selection layer.
            m2 = new_modules[layer_id + 1]
            m2.indexes.data.zero_()
            m2.indexes.data[idx1.tolist()] = 1.0

            layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):
                end_mask = cfg_mask[layer_id_in_cfg]
            continue

    elif isinstance(m0, nn.Conv2d):
        if first_conv:
            # We don't change the first convolution layer.
            m1.weight.data = m0.weight.data.clone()
            first_conv = False
            continue
        if isinstance(old_modules[layer_id - 1], channel_selection):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))

            # If the last layer is channel selection layer, then we don't change the number of output channels of the current
            # convolutional layer.
            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
            m1.weight.data = w1.clone()
            continue

    elif isinstance(m0, nn.Linear):
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))

        m1.weight.data = m0.weight.data[:, idx0].clone()
        m1.bias.data = m0.bias.data.clone()

torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, os.path.join(args.save, 'dense_pruned.pth.tar'))

#print(newmodel)


#use test model to test the accuracy of model after pruning
#test(model)


print("old model time is {}".format(time_old))
a0=time.time()
predict1=newmodel(data0)
a1=time.time()
time_new= a1-a0
print("new model time is {}".format(time_new))
print("speed improved is {}".format(time_new/time_old))


flop, para = profile(newmodel,inputs=(input, ))
print("%.2fM" % (flop/1e6), "%.2fM" % (para/1e6))

#print(newmodel)
#test(newmodel)
