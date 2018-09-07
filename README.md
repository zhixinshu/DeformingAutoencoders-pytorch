# DeformingAutoencoders-pytorch
Pytorch code for DAE, and IntrinsicDAE

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
