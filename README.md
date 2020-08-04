# DeepPixFace-PyTorch

A PyTorch implementation of [Deep Pixel-wise Binary Supervision for Face Presentation Attack Detection](https://arxiv.org/abs/1907.04047)

## Table of Contents:

- [CondenseNet-PyTorch](#condensenet-pytorch)
    - [Data](#data)
    - [Model](#model)
    - [Experiment configs](#experiment-configs)
    - [Requirements](#requirements)
    - [Usage](#usage)
    - [Results](#results)
    - [Future Work](#future-work)
    - [References](#references)
    - [License](#license)


### Data:
Dataloader is responsible for downloading (first time only) and preparing cifar10 data. 

### Model:
To be able to reproduce the results from the official implementation, we use the default model of cifar10 and its configs as given [here](https://github.com/ShichenLiu/CondenseNet).

### Experiment configs:
```
- Input size: 224 x 224 x 3
- Batch size: 16
- Learning rate: 1e-4
- Optimizer: Adam
```
### Usage:
- Clone the repository [here](https://github.com/abhirupkamath/DeepPixFace/master/config.py)
- ``` python main.py ```
- To run on a GPU, you need to enable cuda in the config file.

### Results:
| Metric       | Reproduced  | Official    |
| ------------ |:-----------:|:-----------:|
| Top1 error   |    4.78%    |             |
| Top5 error   |    0.15%    |             |

### Requirements:
Check [requirements.txt](https://github.com/abhirupkamath/DeepPixFace/master/requirements.txt).

### Future Work:
* Add visualization using TensboardX
* Implement the prediction function and demo. 

### References:
* DeepPixBis Official Implementation: https://bit.ly/31kfLY7

### License:
This project is licensed under MIT License - see the LICENSE file for details.
