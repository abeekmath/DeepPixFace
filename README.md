# DeepPixFace-PyTorch

A PyTorch implementation of [Deep Pixel-wise Binary Supervision for Face Presentation Attack Detection](https://arxiv.org/abs/1907.04047) (ICB'19).

## Table of Contents:

- [DeepPixFace-PyTorch](#deeppixface-pytorch)
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
- CASIA Face-Antispoofing dataset was used for training and inference.
- Can be downloaded from [here](https://competitions.codalab.org/competitions/22036#learn_the_details).
    - Requires signing and mailing agreement to competition organizers.
- Research paper for the [CASIA-SURF Dataset](https://arxiv.org/pdf/1908.10654.pdf)

### Model:
- DenseNet based implementation with dual supervision. 

### Experiment configs:
```
- Input size: 224 x 224 x 3
- Batch size: 8
- Learning rate: 1e-4
- Optimizer: Adam
- Model was trained locally on a GTX 1050Ti. 
- Small batch-size of 8 is used to ensure model meets local compute requirements.
```
### Usage:
- Clone the repository [here](https://github.com/abhirupkamath/DeepPixFace/blob/master/config.py)
- ``` python main.py ```
- To run on a GPU, you need to enable cuda in the config file.

### Future-Work:
- Update repo with demo visualizations corresponding to implementation 

### Requirements:
Check [requirements.txt](https://github.com/abhirupkamath/DeepPixFace/blob/master/requirements.txt).

### References:
* DeepPixBis Official Implementation: https://bit.ly/31kfLY7

### License:
This project is licensed under Apache-2.0 License - see the LICENSE file for details.
