# DeformingAutoencoders-pytorch
Pytorch code for DAE and IntrinsicDAE

# Project:
http://www3.cs.stonybrook.edu/~cvl/dae.html

# Usage:
Requirements: PyTorch

To train a DAE, run

python train_DAE_CelebA.py --dirDataroot=[path_to_root_of_training_data] --dirCheckpoints=[path_to_checkpoints] --dirImageoutput=[path_to_output directory for training] --dirTestingoutput=[path_to_output directory for testing]

To train an IntrinsicDAE, run

python train_IntrinsicDAE_CelebA.py --dirDataroot=[path_to_root_of_training_data] --dirCheckpoints=[path_to_checkpoints] --dirImageoutput=[path_to_output directory for training] --dirTestingoutput=[path_to_output directory for testing]

set --useDense=True (default) for DenseNet-like encoder/decoder (no skip connections over the bottleneck latent representations); --useDense=False for a smaller encoder-decoder architecture.

Dataset: 
CelebA (http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
A google drive link to a cropped and resized version of CelebA: 
https://drive.google.com/open?id=1ueB8BJxid2rZbvh3RaoZ9lDdlKH4B-pL

Place the training images in [path_to_root_of_training_data]/celeba_split/img_00
(Split the dataset into multiple subsets if wanted.)

Checkpoints:
Some example checkpoints can be found at:
https://drive.google.com/drive/folders/1A2Qj1NhzVU5XSjeilKhjWwAgNWvlRyuA?usp=sharing
Three examples are provided:
1. DAE for CelebA with Dense encoder decoder, where opt.idim = 8 (./DAE_CelebA_idim8)
2. DAE for CelebA with Dense encoder decoder, where opt.idim = 16 (./DAE_CelebA_idim16)
3. IntrinsicDAE for CelebA with Dense encoder decoder, where opt.idim = 32 (./IntrinsicDAE_CelebA)

If using the code, please cite: 

Deforming Autoencoders: Unsupervised Disentangling of Shape and Appearance, Zhixin Shu, Mihir Sahasrabudhe, Riza Alp Guler, Dimitris Samaras, Nikos Paragios, and Iasonas Kokkinos. European Conference on Computer Vision (ECCV), 2018.

