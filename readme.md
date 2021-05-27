# Using Mean Shift Network Pruning in TEE to Improve DNN's Model Inference Speed

## Acknowledgement: 

The Alpine-Pytorch Dockerfile used here are referenced form [SGX-LKL](https://github.com/lsds/sgx-lkl).
Some network pruning code is used from the [Network Slimming](https://github.com/Eric-mingjie/network-slimming).
To run our model in TEE we choose [Occlum](https://github.com/occlum/occlum) SGX runtime.

## Prune

```
python vggprune.py
python resprune.py
python denseprune.py
```
The pruned model will be named [modelname].pth.tar.

## Fine-tune
```angular2html
python main.py --refine weights/vgg_pruned.pth.tar --dataset cifar10 --arch vgg --depth 16
python main.py --refine weights/res_pruned.pth.tar --dataset cifar10 --arch resnet --depth 164
python main.py --refine weights/dense_pruned.pth.tar --dataset cifar10 --arch densenet --depth 40
```

##Test Inference Speed
```angular2html
python test.py --dataset cifar10 --arch densenet --depth 40
python test.py --test weights/dense_pruned.pth.tar --dataset cifar10 --arch densenet --depth 40
```
##Things to improve
Runing pytorch in occlum may not working, I still need to working on this part to provide a script allow public use this project easier. But this project has been test on my own machine and all the result wrote in the thesis is obtian from my experinment. Runing pruning on native linux environment is absolutely fine and the inference speed up is obverse.