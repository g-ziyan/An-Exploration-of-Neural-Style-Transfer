### Introduction
This repo is for my course project of EECE570. Neural Style Transfer (NST) is one of the hottest topics in 
computer vision in recent years, which studies on how to use a Convolutional Neural Networks to reproduce 
famous painting styles (style images) on natural images (content images). This project aims to verify the applicability
of NST on a wider variety of styles. My implementation is based on Johnson et al.'s algorithm which combines perceptual loss
and feed-forward networks. I extended the model by modifying the structures of instance normailzaiton, up-convolutional
layer and reconstruction layer. I also added the feature of color preserving. 

### Results

![result](https://github.com/g-ziyan/An_Exploration_of_Neural_Style_Transfer/blob/master/imgs/result.png)

![result](https://github.com/g-ziyan/An_Exploration_of_Neural_Style_Transfer/blob/master/imgs/result2.png)

### Reference
1. Torch (Lua) implementation by [jcjohnson](https://github.com/jcjohnson/fast-neural-style)
2. PyTorch implementation by [pytorch](https://github.com/pytorch/examples/tree/master/fast_neural_style)
