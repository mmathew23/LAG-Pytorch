# LAG-Pytorch

Unofficial Pytorch Implementation for the paper: 
"[Creating High Resolution Images with a Latent Adversarial Generator](https://arxiv.org/abs/2003.02365)"
by David Berthelot, Peyman Milanfar, and Ian Goodfellow.

The official tensorflow implementation was used as reference
https://github.com/google-research/lag/

I've deviated in some places from the paper. Most notably I do not use the progressive training strategy.


## Training a Model
To start training run the below command  
`
python train.py --model_name_prefix test_run --root_path /path_to_data --batch_size 8 --upsample_layers 3  
`

