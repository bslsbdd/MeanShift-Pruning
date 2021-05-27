import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from models import *
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import KMeans
import time
import matplotlib.pyplot as plt
from itertools import cycle
from thop import profile

parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--depth', type=int, default=16,
                    help='depth of the vgg')
parser.add_argument('--model', default='model_best.pth.tar', type=str, metavar='PATH',
                    help='path to the model (default: none)')
parser.add_argument('--save', default='weights/', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if not os.path.exists(args.save):
    os.makedirs(args.save)

model = vgg(dataset=args.dataset, depth=args.depth)
if args.cuda:
    model.cuda()

if args.model:
    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        checkpoint = torch.load(args.model,map_location=torch.device('cpu'))
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.model, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
print(model)

def test(model):
    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'cifar10':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data.cifar10/test_batch', train=False,download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    elif args.dataset == 'cifar100':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    else:
        raise ValueError("No valid dataset is given.")
    model.eval()
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1] 
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(
        correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))

#test(model)

cc = []
for k,m in enumerate(model.modules()):
    if isinstance(m, nn.BatchNorm2d):
        cc.append(m.weight.data)
        print(k)

ccc = []
for i in cc:
    i = i.numpy()
    i = i.tolist()
    ccc.append(i)
    
thre_all = []
kk = []
for ii in range(4):
    if ii == 0:
        ww = ccc[0]+ccc[1]+ccc[2]
        ww = np.array(ww)
        ww = np.reshape(ww, (-1,1))
        bandwidth = estimate_bandwidth(ww, quantile=0.1, n_samples=len(ww))
        meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        meanshift.fit(ww)
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
        thre_all.append(thre)
       
    elif ii == 1:
        ww1 = ccc[3]+ccc[4]+ccc[5]
        ww1 = np.array(ww1)
        ww1 = np.reshape(ww1, (-1,1))
        bandwidth = estimate_bandwidth(ww1, quantile=0.1, n_samples=len(ww1))
        meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        meanshift.fit(ww1)
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
        #thre = centroids[1][0]
        thre_all.append(thre)
        
    elif ii == 2:
        ww2 = ccc[6]+ccc[7]+ccc[8]
        ww2 = np.array(ww2)
        ww2 = np.reshape(ww2, (-1,1))
        bandwidth = estimate_bandwidth(ww2, quantile=0.1, n_samples=len(ww2))
        meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        meanshift.fit(ww2)
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
        #thre = centroids[1][0]
        thre_all.append(thre)
        
    elif ii == 3:
        ww3 = ccc[9]+ccc[10]+ccc[11]+ccc[12]
        ww3 = np.array(ww3)
        ww3 = np.reshape(ww3, (-1,1))   
        bandwidth = estimate_bandwidth(ww3, quantile=0.1, n_samples=len(ww3))
        meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        meanshift.fit(ww3)
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
        #thre = centroids[1][0]
        thre_all.append(thre)        

cfg = []
cfg_mask = []
for k, m in enumerate(model.modules()):
    if isinstance(m, nn.BatchNorm2d):
        pruned = 0
        if k == 3:thre = thre_all[0]
        elif k== 6:thre = thre_all[0]
        elif k== 10:thre = thre_all[0]
        elif k== 13:thre = thre_all[1]
        elif k== 17:thre = thre_all[1]
        elif k== 20:thre = thre_all[1]
        elif k== 23:thre = thre_all[2]
        elif k== 27:thre = thre_all[2]
        elif k== 30:thre = thre_all[2]
        elif k== 33:thre = thre_all[3]
        elif k== 37:thre = thre_all[3]
        elif k== 40:thre = thre_all[3]
        elif k== 43:thre = thre_all[3]
        size = m.weight.data.shape[0]
        weight_copy = m.weight.data.abs().clone()
        mask = weight_copy.gt(thre).float()
        pruned = pruned + mask.shape[0] - torch.sum(mask)
        m.weight.data.mul_(mask)
        m.bias.data.mul_(mask)
        cfg.append(int(torch.sum(mask)))
        cfg_mask.append(mask.clone())
        pruned_ratio = pruned/size
        print('k:',pruned_ratio)
        print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
              format(k, mask.shape[0], int(torch.sum(mask))))
    elif isinstance(m, nn.MaxPool2d):
        cfg.append('M')
     
print('Pre-processing Successful!')




print(cfg)
newmodel = vgg(dataset=args.dataset, cfg=cfg)
if args.cuda:
    newmodel.cuda()

num_parameters = sum([param.nelement() for param in newmodel.parameters()])
savepath = os.path.join(args.save, "prune.txt")
with open(savepath, "w") as fp:
    fp.write("Configuration: \n"+str(cfg)+"\n")
    fp.write("Number of parameters: \n"+str(num_parameters)+"\n")
    #fp.write("Test accuracy: \n"+str(acc))

layer_id_in_cfg = 0
start_mask = torch.ones(3)
end_mask = cfg_mask[layer_id_in_cfg]
for [m0, m1] in zip(model.modules(), newmodel.modules()):
    if isinstance(m0, nn.BatchNorm2d):
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        if idx1.size == 1:
            idx1 = np.resize(idx1,(1,))
        m1.weight.data = m0.weight.data[idx1.tolist()].clone()
        m1.bias.data = m0.bias.data[idx1.tolist()].clone()
        m1.running_mean = m0.running_mean[idx1.tolist()].clone()
        m1.running_var = m0.running_var[idx1.tolist()].clone()
        layer_id_in_cfg += 1
        start_mask = end_mask.clone()
        if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
            end_mask = cfg_mask[layer_id_in_cfg]
    elif isinstance(m0, nn.Conv2d):
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))
        if idx1.size == 1:
            idx1 = np.resize(idx1, (1,))
        w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
        w1 = w1[idx1.tolist(), :, :, :].clone()
        m1.weight.data = w1.clone()
    elif isinstance(m0, nn.Linear):
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))
        m1.weight.data = m0.weight.data[:, idx0].clone()
        m1.bias.data = m0.bias.data.clone()

torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, os.path.join(args.save, 'vgg_pruned.pth.tar'))


"""
#print(newmodel)
data0 = torch.randn(1*3,32*32)
data0 = data0.reshape(1,3,32,32)
a0=time.time()
predict0=model(data0)
a1=time.time()
time_old = a1-a0
print("old model time is {}".format(time_old))
a0=time.time()
predict1=newmodel(data0)
a1=time.time()
time_new= a1-a0
print("new model time is {}".format(time_new))
print("speed improved is {}".format(time_new/time_old))


input = torch.randn(1, 3, 32, 32)
flop, para = profile(model, inputs=(input,))
print("%.2fM" % (flop/1e6), "%.2fM" % (para/1e6))
flop, para = profile(newmodel, inputs=(input,))
print("%.2fM" % (flop/1e6), "%.2fM" % (para/1e6))
"""
#test(model)
#test(newmodel)
#test(model)
#test(model)

