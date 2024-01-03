# artemis_treescaling
Individual-tree attribute scaling from airborne laser scanning point clouds

![graphic_abstract](https://github.com/truebelief/artemis_treescaling/assets/8785889/16166503-b2fb-480c-bf6a-0a6bf2e8ebf4)


![Figure02](https://github.com/truebelief/artemis_treescaling/assets/8785889/bae5fcf4-f73f-41d9-95d0-16d73540f814)

## Citation
Training strategy:

<cite> Xi, Z.; Hopkinson, C.; Chasmer, L. Filtering Stems and Branches from Terrestrial Laser Scanning Point Clouds Using Deep 3-D Fully Convolutional Networks. Remote Sens. 2018, 10, 1215. https://doi.org/10.3390/rs10081215.</cite>


SegFormer:

<cite>Xie, Enze, et al. "SegFormer: Simple and efficient design for semantic segmentation with transformers." Advances in Neural Information Processing Systems 34 (2021): 12077-12090.</cite>


SegFormer3D:

<cite>Xi, Zhouxin, Laura Chasmer, and Chris Hopkinson. "Delineating and Reconstructing 3D Forest Fuel Components and Volumes with Terrestrial Laser Scanning." Remote Sensing 15.19 (2023): 4778.</cite>

## Folder structure

    ├── vegcls                       # 1. Project: Above-ground tree point classification from ALS using deep learning
    │   ├── data                     # Customizable ALS dataset for training, validation, and testing: Manually labelled ALS reference point clouds (*.laz/*.las format only now); the pointwise labels are two classes (other:1,tree:2), stored as the extra bytes field of laz file 
    │   │   ├── apply                # Customizable ALS dataset for applying the trained model (just a subset from ALS tiles due to the file size limit in GitHub)
    │   ├── config                   # Configuration files
    │   │   ├── config.json          # Update your data paths and network structure parameters here (you can also choose either SegFormer(default) or ResNet50 by (un-)commenting the model types)
    │   │   ├── train_list.txt       # Customizable training file names
    │   │   ├── valid_list.txt       # Customizable validating file names (validating step occurs every N iterations during the training process)
    │   │   ├── test_list.txt        # Customizable testing file names (testing step independent from the training process)
    │   │   ├── test_list.txt        # Customizable applicating file names (apply the trained model for prediction over larger areas)
    │   ├── code                     
    │   │   ├── vegclsMain.py        # Main entry; you can customize the mode (train/test/apply)
    │   │   ├── vox3DSegFormer.py    # SegFormer network layer definition
    │   │   ├── vox3DResNetModel.py  # ResNet network layer definition
    │   ├── logs                     # This is a generated folder containing all logs and outcomes from training to application steps
    ├── LICENSE
    └── README.md


## 1. Classification of above-ground tree points from ALS using deep learning

<img src='https://github.com/truebelief/artemis_treescaling/assets/8785889/59bd9f99-abf5-4f80-bc5a-ea623342dd34' width=800>


**1.1** Install python (3.9+)

**1.2** Install PyTorch following the official guide from:
https://pytorch.org/

For example using pip:
`python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

Currently, I use the most recent PyTorch 2.0+ (Windows) without any problem; I could also run the program using PyTorch 1.12+ years ago. I haven't tested the programs in other environments yet. If you find any compatibility issues, please let me know.

**1.3** In addition, install necessary Python libraries:
`python -m pip install numpy numpy-indexed numpy_groupies commentjson laspy[lazrs] timm`

*I find it frustrating when programmers overuse dependencies for mere convenience, rather than for functional efficiency. It's also bothersome when they focus more on showcasing their coding skills through excessive code refactorization, but are blind to practical demands and real-world challenges.

**1.4** Customize the data folder and config.json files on your own

**1.5** Run python:
You can use python programming IDE or directly use the command

training:
`python yourfolder/vegcls/code/vegclsMain.py --mode train`

testing:
`python yourfolder/vegcls/code/vegclsMain.py --mode test`

testing:
`python yourfolder/vegcls/code/vegclsMain.py --mode apply`

**1.5** Results will be exported to the logs folder
including a log file and best-trained models

### Benchmarking

<img src='https://github.com/truebelief/artemis_treescaling/assets/8785889/d707f0cd-e0df-4ce1-8e2c-f4ec4497d113' width=600>



