## Meta-learning for Solving Inverse Problem
* meta_module.py, modules.py, utils.py, dataio.py, inner_modules.py are essential scripts for proposed method. 
* Note: to run the scripts, one need to correct the path informatin in dataio.py with path to local CelebA dataset. (such correction may need to be done at many places, like denoiser/denoiser.py, etc.)

## Inpainting task
* script_inpainting.py is sample experiment script for training for inpainting task
* eval_inpainting.py is sample evaluation script 
* model files used are stored in model_zoo/inpainting_center, and model_zoo/inpainting_random
* UNet implementation can be found in UNet folder

## Deconvolution task
* denoiser training code can be found in denoiser/, with trained denoiser stored in model_zoo/denoiser/dncnn.ckpt
* script_deblur_meta_fp.py is sample experiment script for training for inpainting task
* eval_deblur_ircnn_fp.py is sample evaluation script 
* model files used are stored in model_zoo/deblur
* HQS related file in DPIR folder (both training and eval)
* red_fp.py is the evaluation script for RED (training of unroll RED can be done by slightly modify script_deblur_meta_fp.py)

## Reference
* more reference can be found in the writeup
* a lots of reference are taken from: https://github.com/vsitzmann/siren
, https://github.com/cszn/DPIR, https://github.com/vsitzmann/pytorch_prototyping
