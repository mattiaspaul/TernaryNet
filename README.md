# TernaryNet

pytorch implementation for the paper

Mattias P. Heinrich, Max Blendowski, Ozan Oktay
"TernaryNet: Faster Deep Model Inference without GPUs for Medical 3D Segmentation using Sparse and Binary Convolutions"
currently under review for IJCARS MICCAI 2017 special issue

see https://arxiv.org/abs/1801.09449

Currently, only the most basic training/validation example using ternary convolutions within a U-Net medical image segmentation pipeline are provided. This will be extended in the near future, also with the addition of Hamming distance optimised C-code for inference.

``
    m = torch.nn.Tanh()
    y = m((x*beta*2.0-beta))*0.5
    y += -m((-x*beta*2.0-beta))*0.5
``

If you find the material useful please cite the above paper or contact me through my website mpheinrich.de

