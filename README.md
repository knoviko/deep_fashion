# Running
____
Train pre-trained model using "Deep Fashion In-store" dataset: https://drive.google.com/drive/folders/0B7EVK8r0v71pYkd5TzBiclMzR00?resourcekey=0-fsjVShvqXP2517KnwaZ0zw  
Dataset should be stored in the */DeepFashion/* folder, only *"img.zip"* archive is needed.  
The format of the train-listfile, val-listfile (DeepFashion/val-listfile.json) files is as follows:  
```
{
   "images": [
    "/home/knoviko/projects/DeepFashion/img/MEN/Pants/id_00006830/08_2_side.jpg",
    "/home/knoviko/projects/DeepFashion/img/WOMEN/Tees_Tanks/id_00005837/07_1_front.jpg",
    "/home/knoviko/projects/DeepFashion/img/WOMEN/Skirts/id_00005949/05_3_back.jpg",
    "/home/knoviko/projects/DeepFashion/img/WOMEN/Graphic_Tees/id_00000917/01_7_additional.jpg"
],
   "categories": [
      "Pants",
      "Tees_Tanks",
      "Skirts",
      "Graphic_Tees"
   ]
 }
```
For initiating model training on the Deep Fashion In-store dataset, execute the following command

  *sbatch ./hierarchicalContrastiveLearning/df_job.sh*
  
*df_job.sh* file (hierarchicalContrastiveLearning/df_job.sh): 
```python
#!/bin/bash
#SBATCH --job-name=deep_fashion_task            # Name of the job
#SBATCH -t 0-07:00:00                           # Max running time
##SBATCH --mem-per-cpu=1500MB                   # Define memory per node  # this one is not in effect, due to the double hash
#SBATCH --mem=64GB                              # Define memory per CPU/core
#SBATCH --gpus=V100:2                           # request 1 Volta V100 GPU (Number and type of the GPU)
#SBATCH --partition=informatik-mind             # Define the partition on which the job shall run
#SBATCH -n 5                                    # use 5 tasks
#SBATCH --cpus-per-task=1                       # use 1 thread per taks (Number of the CPU for per task)
#SBATCH -N 1                                    # request slots on 1 node 
#SBATCH --error=deep_fashion_task-%j.err        # Error-outpt file 
#SBATCH --output=deep_fashion_task-%j.log       # Result-output file
##SBATCH --mail-type=END,FAIL,TIME_LIMIT         # send notification emails

# Loading Anaconda module
module load anaconda3/latest
. $ANACONDA_HOME/etc/profile.d/conda.sh

export PYTHONPATH="/home/knoviko/projects/hierarchicalContrastiveLearning"
conda activate env_df1
python ./hierarchicalContrastiveLearning/classification/train_deepfashion.py --data /home/knoviko/projects/ --train-listfile DeepFashion/train_listfile.json --val-listfile DeepFashion/val_listfile.json --class-map-file DeepFashion/class_map.json --repeating-product-file DeepFashion/repeating_product_ids.csv --num-classes 17 --epochs 1 --learning_rate 0.5 --temp 0.1 --ckpt /home/knoviko/projects/hierarchicalContrastiveLearning/pretrained_model/resnet50-19c8e357.pth --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --cosine
#python ./hierarchicalContrastiveLearning/classification/eval_deepfashion.py --data /home/knoviko/projects/ --train-listfile DeepFashion/train_listfile.json --val-listfile DeepFashion/val_listfile.json --test-listfile DeepFashion/test_listfile.json --class-map-file DeepFashion/class_map.json --repeating-product-file DeepFashion/repeating_product_ids.csv --num-classes 17 --epochs 1 --learning_rate 0.5 --ckpt /home/knoviko/projects/model/hmlc_dataset_resnet50_lr_0.5_decay_0.1_bsz_512_loss_hmce_trial_5/checkpoint_0001.pth.tar  

conda deactivate
```
  To evaluate the model, run the same file *df_job.sh* by uncommenting the line:
```
python ./hierarchicalContrastiveLearning/classification/eval_deepfashion.py --data /home/knoviko/projects/ --train-listfile DeepFashion/train_listfile.json --val-listfile DeepFashion/val_listfile.json --test-listfile DeepFashion/test_listfile.json --class-map-file DeepFashion/class_map.json --repeating-product-file DeepFashion/repeating_product_ids.csv --num-classes 17 --epochs 1 --learning_rate 0.5 --ckpt /home/knoviko/projects/model/hmlc_dataset_resnet50_lr_0.5_decay_0.1_bsz_512_loss_hmce_trial_5/checkpoint_0001.pth.tar
```  

____
  Source repository from the article:
https://github.com/salesforce/hierarchicalContrastiveLearning/tree/master
