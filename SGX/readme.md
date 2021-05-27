## Use  Pruned Model with Occlum

To use Occlum first we need build Occlum Docker images. Follow this [page](https://github.com/occlum/occlum/tree/master/tools/docker) to build Occlum Docker images.
To use Pytorch in Occlum please follow the instruction I wrote [here](https://github.com/chengyaox/occlum/tree/master/demos/pytorch).

## Things to take care of

1. Use the DockerFile and script provide here.
2. Copy the test.py and pruned model in to the occlum workspace ``image/bin``.
3. When runing _run_pytorch_on_occlum.sh_ some library may not exist that is ok.
4. Replace the _Occlum.json_ file in occlum instance by file provide here.
5. Remember we need use ```SGX_MODE=SIM occlum build``` to build occlum instance every time we change some file.

## Test the model inference speed
```angular2html
occlum run /bin/python3.7 /bin/[path to test.py] --dataset cifar10 --arch vgg --depth 16         #baseline
occlum run /bin/python3.7 /bin/[path to test.py] --test [path to pruned model] --dataset cifar10 --arch vgg --depth 16
occlum run /bin/python3.7 /bin/[path to test.py] --dataset cifar10 --arch resnet --depth 164     #baseline
occlum run /bin/python3.7 /bin/[path to test.py] --test [path to pruned model] --dataset cifar10 --arch resnet --depth 164
occlum run /bin/python3.7 /bin/[path to test.py] --dataset cifar10 --arch densenet --depth 40    #baseline
occlum run /bin/python3.7 /bin/[path to test.py] --test [path to pruned model] --dataset cifar10 --arch densenet --depth 40

```

## Things to improve
Runing pytorch in occlum may not working, I still need to working on this part to provide a script allow public use this project easily. But this project has been test on my own machine and all the result wrote in the thesis is obtian from my experinment.
