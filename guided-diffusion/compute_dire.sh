## set MODEL_PATH, num_samples, has_subfolder, images_dir, recons_dir, dire_dir
export CUDA_VISIBLE_DEVICES=0
export NCCL_P2P_DISABLE=1
 export  RDMAV_FORK_SAFE=0
MODEL_PATH="/home/lorenzp/DeepFakeDetectors/DIRE/256x256_diffusion_uncond.pt" 

SAMPLE_FLAGS="--batch_size 1 --num_samples 1  --timestep_respacing ddim20 --use_ddim True"
SAVE_FLAGS="--images_dir /home/lorenzp/DeepFakeDetectors/DIRE/data/adversarial/images --recons_dir /home/lorenzp/DeepFakeDetectors/DIRE/data/adversarial/recons --dire_dir /home/lorenzp/DeepFakeDetectors/DIRE/data/adversarial/dire"
# SAVE_FLAGS="--images_dir /home/lorenzp/DeepFakeDetectors/DIRE/data/adversarial/nor --recons_dir /home/lorenzp/DeepFakeDetectors/DIRE/data/adversarial/recons --dire_dir /home/lorenzp/DeepFakeDetectors/DIRE/data/adversarial/dire"
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
mpiexec -n 1 python compute_dire.py --model_path $MODEL_PATH $MODEL_FLAGS  $SAVE_FLAGS $SAMPLE_FLAGS --has_subfolder False