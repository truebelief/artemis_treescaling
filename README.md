# Individual-tree detection, segmentation, and attribute regression from airborne laser scanning point clouds

The University of Lethbridge - Department of Geography & Environment - Artemis Lab

Author - Zhouxin Xi (zhouxin.xi@uleth.ca), Chris Hopkinson (c.hopkinson@uleth.ca), and Laura Chasmer (laura.chasmer@uleth.ca)

<img src='https://github.com/truebelief/artemis_treescaling/assets/8785889/16166503-b2fb-480c-bf6a-0a6bf2e8ebf4' width=1000>

<img src='https://github.com/truebelief/artemis_treescaling/assets/8785889/bae5fcf4-f73f-41d9-95d0-16d73540f814' width=1000>

Note: The algorithms presented here were primarily developed between 2020 and 2022. While they may not represent the latest advancements in AI for achieving optimal accuracy, given the rapid progress in the field, they might offer practical and thought-provoking approaches for your own study.


## Please cite

Main publication:

Xi, Zhouxin, Chris Hopkinson, and Laura Chasmer. "Supervised terrestrial to airborne laser scanner model calibration for 3D individual-tree attribute mapping using deep neural networks." ISPRS Journal of Photogrammetry and Remote Sensing 209 (2024): 324-343. [link](https://doi.org/10.1016/j.isprsjprs.2024.02.010).

Training strategy:

<cite> Xi, Zhouxin, Chris Hopkinson, and Laura Chasmer. "Filtering stems and branches from terrestrial laser scanning point clouds using deep 3-D fully convolutional networks." Remote Sensing 10.8 (2018): 1215. [link](https://doi.org/10.3390/rs10081215).</cite>


SegFormer:

<cite> Xie, Enze, et al. "SegFormer: Simple and efficient design for semantic segmentation with transformers." Advances in Neural Information Processing Systems 34 (2021): 12077-12090.</cite>


SegFormer3D:

<cite> Xi, Zhouxin, Laura Chasmer, and Chris Hopkinson. "Delineating and Reconstructing 3D Forest Fuel Components and Volumes with Terrestrial Laser Scanning." Remote Sensing 15.19 (2023): 4778. [link](https://doi.org/10.3390/rs15194778). </cite>

Cut-pursuit:

<cite> Landrieu, Loic, and Guillaume Obozinski. "Cut pursuit: Fast algorithms to learn piecewise constant functions on general weighted graphs." SIAM Journal on Imaging Sciences 10.4 (2017): 1724-1766. [link](https://hal.archives-ouvertes.fr/hal-01306779)</cite>

## Folder structure

    ├── vegcls                       # 1. Project: 3D tree point classification from ALS using deep learning
    │   ├── data                     # Customizable ALS dataset for training, validation, and testing: Manually labelled ALS reference point clouds (*.laz/*.las format only now); the pointwise labels are two classes (other:1,tree:2), stored as the extra bytes field of laz file 
    │   │   ├── apply                # Customizable ALS dataset for applying the trained model (just a subset from ALS tiles due to the file size limit in GitHub)
    │   ├── config                   # Configuration files
    │   │   ├── config.json          # Update your data paths and network structure parameters here (you can also choose either SegFormer (default) or ResNet50 by (un-)commenting the model types)
    │   │   ├── train_list.txt       # Customizable training file names
    │   │   ├── valid_list.txt       # Customizable validating file names (validating step occurs every N iterations during the training process)
    │   │   ├── test_list.txt        # Customizable testing file names (testing step independent from the training process)
    │   │   ├── app_list.txt         # Customizable applicating file names (apply the trained model for prediction over larger areas)
    │   ├── code                     
    │   │   ├── vegclsMain.py        # Main entry; you can customize the mode (train/test/apply)
    │   │   ├── vox3DSegFormer.py    # SegFormer network layer definition
    │   │   ├── vox3DResNetModel.py  # ResNet network layer definition
    │   ├── logs                     # This is a generated folder containing all logs and outcomes from training to application steps
    |
    ├── itcseg                       # 2. Project: 3D individual-tree instance segmentation from ALS
    │   ├── data                     # Customizable ALS dataset for training, validation, and testing
    │   │   ├── apply                # Customizable ALS dataset for applying the trained model
    │   ├── config                   # Configuration files
    │   │   ├── config.json          # Update your data paths and network structure parameters here
    │   │   ├── train_list.txt       # Customizable training file names
    │   │   ├── valid_list.txt       # Customizable validating file names
    │   │   ├── test_list.txt        # Customizable testing file names
    │   │   ├── app_list.txt         # Customizable applicating file names
    │   ├── code                     
    │   │   ├── itcsegMain.py        # Main entry; you can customize the mode (train/test/apply)
    │   │   ├── vox3DSegFormer.py    # SegFormer network layer definition
    │   │   ├── itcsegPost.m         # Matlab post-processing that clusters remaining points to the tree center regions detected from the itcsegMain.py
    │   │   ├── cutPursuit.m         # Cut-pursuit clustering algorithm 
    │   │   ├── \*.mexw64            # Compiled binary files for cut-pursuit and laz file I/O functions	
    │   ├── logs                     # This is a generated folder containing all logs and outcomes from training to application steps
    |
    ├── itcreg                       # 3. Project: Individual-tree level regression using simple deep-learning modules
    │   ├── data                     # Customizable ALS dataset for training, validation, and testing
    │   │   ├── apply                # Customizable ALS dataset for applying the trained model
    │   ├── config                   # Configuration files
    │   │   ├── config.json          # Update your data paths and network structure parameters here
    │   │   ├── train_list.txt       # Customizable training file names
    │   │   ├── valid_list.txt       # Customizable validating file names
    │   │   ├── test_list.txt        # Customizable testing file names
    │   │   ├── app_list.txt         # Customizable applicating file names
    │   ├── code                     
    │   │   ├── itcregMain.py        # Main entry; you can customize the mode (train/test/apply)
    │   ├── logs                     # This is a generated folder containing all logs and outcomes from training to application steps
    |
    ├── LICENSE
    └── README.md


## 1. Requirement

Requires a CUDA-compatible GPU with more than 8GB VRAM. Tested on an RTX3090.

For GPUs with less VRAM, consider lowering the voxel numbers per block in the configuration file to accommodate.

<details>
	<summary>
		1.1 Install python (3.9+)
	</summary>
</details>

<details>
	<summary>
		1.2 Install PyTorch following the official guide from:&nbsp
	</summary>
<br />
	
https://pytorch.org/


For example, using pip:

`python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

Currently, I use the most recent PyTorch 2.0+ (Windows) without any problem; I could also run the program using PyTorch 1.12+ years ago. I haven't tested the programs in other environments yet. If you find any compatibility issues, please let me know.

</details>

<details>
	<summary>
		1.3 Install the additional Python libraries:
	</summary>
<br />
	
`python -m pip install numpy numpy-indexed numpy_groupies commentjson laspy[lazrs] timm tqdm gpytorch`


*I find it frustrating when programmers overuse dependencies for mere convenience, rather than for functional efficiency. It's also bothersome when they focus more on showcasing their coding skills through excessive code refactorization, but are blind to practical demands and real-world challenges.

</details>

## 2. Classification of above-ground tree points from ALS using deep learning
<p align="center">
<img src='https://github.com/truebelief/artemis_treescaling/assets/8785889/59bd9f99-abf5-4f80-bc5a-ea623342dd34' width=800>
</p>


<details>
	<summary>
		2.1 Customize the data folder and config.json files on your own
	</summary>
</details>

<details>
	<summary>
		2.2 Run Python
	</summary>
	
<br />

You can use Python programming IDE or directly use the command line

* training:

`python yourfolder/vegcls/code/vegclsMain.py --mode train`

* testing:

`python yourfolder/vegcls/code/vegclsMain.py --mode test`

* application (only after the model is trained):

`python yourfolder/vegcls/code/vegclsMain.py --mode apply`

</details>


<details>
	<summary>
		2.3 Results will be exported to the "logs" folder where:
	</summary>
<br />

"logs/train" includes a log file and the best-trained model

"logs/app" includes a log file and the prediction of per-point classes (2: vegetation/tree, 1:ground and other points). The class will be saved into the output laz file as the extra-byte scalar field "VegCls".

</details>

You can apply the trained model to a much larger area, e.g., the landscape level as below:

<p align="center">
    <img src='https://github.com/truebelief/artemis_treescaling/assets/8785889/ae10462d-b80c-48b0-9b06-07dac4a23dc0' width=600>
</p>



### Benchmarking
<p align="center">
    <img src='https://github.com/truebelief/artemis_treescaling/assets/8785889/d707f0cd-e0df-4ce1-8e2c-f4ec4497d113' width=500>
</p>

## 3. Segmentation of individual tree points from ALS

There are two main steps involved: classification of tree center regions, and segmentation of the tree boundaries.


The first step uses the same SegFormer model, identical to the previous classification section

<details>
	<summary>		
		3.1 Customize the data folder and config.json files on your own
	</summary>
	
<br />

The reference dataset (*.laz or *.las format only) should follow those in the data folder, with a specific scalar field "itc_ref" identifying the tree ID for each point, and the scalar field "VegCls" from the previous classification module (2: vegetation/tree, 1:ground and other points).

The application dataset should have the scalar field "VegCls" from the previous classification module  (2: vegetation/tree, 1:ground and other points).

</details>

<details>
	<summary>
		3.2 Run python
	</summary>
<br />

You can use a Python programming IDE or directly use the command line

* prepare:

`python yourfolder/itcseg/code/itcsegMain.py --mode prepare`

* training:

`python yourfolder/itcseg/code/itcsegMain.py --mode train`

* testing:

`python yourfolder/itcseg/code/itcsegMain.py --mode test`

* application (only after the model is trained):

`python yourfolder/itcseg/code/itcsegMain.py --mode apply`

</details>

<details>
	<summary>
		3.3 Results will be exported where:
	</summary>
	
<br />
	
"data" includes the result files ("*_cfd.laz") from the "prepare" step. Smooth confidence for each point of the input point cloud will be created based on the reference dataset, which will be used to guide the training process for tree center detection.

"logs/train" includes a log file and the best-trained model

"logs/app" includes a log file and the prediction of per-point classes (2:tree center, 1:others). The class will be saved into the output laz file as the extra-byte scalar field "ConfPred".

</details>

<details>
	<summary>
		3.4 Run Matlab code (itcsegPost.m):
	</summary>
	
<br />

Please customize the file path of input and output. The input will be the result laz files from "logs/app", and the default output folder is also "logs/app". The output file ends with "_segs.laz"

The cut-pursuit algorithm and laz I/O functions have been compiled into mexw64 binary files for Windows. If you're using a different operating system, you may need to recompile them.

</details>

You can apply the trained model to a much larger area, e.g., the landscape level as below:

<p align="center">
    <img src='https://github.com/truebelief/artemis_treescaling/assets/8785889/1e2ac757-8938-47c1-9476-429c25702229' width=600>
</p>

### Benchmarking
<p align="center">
    <img src='https://github.com/truebelief/artemis_treescaling/assets/8785889/1b77ebd1-61cf-402d-a3f2-a18070bd9f34' width=800>
</p>

## 4. Regression of individual-tree attributes from TLS to ALS
<p align="center">
    <img src='https://github.com/truebelief/artemis_treescaling/assets/8785889/982191b1-d40d-4c36-bed4-153499cbbad9' width=500>
</p>

<details>
	<summary>
		4.1 Customize the data folder and config.json files on your own
	</summary>
	
<br />
	
The reference dataset (*.laz or *.las format only) should follow those in the data folder, with a specific scalar field "itc_ref" identifying the ID for each tree, and "AttrRef" as the reference attribute (Tree diameter-at-breast height as our example).

The application dataset should have the scalar field "segs" from the previous tree segmentation module.

</details>

<details>
	<summary>
		4.2 Run python
	</summary>

<br />

You can use a Python programming IDE or directly use the command line

* training:

`python yourfolder/itcreg/code/itcregMain.py --mode train`

* testing:

`python yourfolder/itcreg/code/itcregMain.py --mode test`

* application (only after the model is trained):

`python yourfolder/itcreg/code/itcregMain.py --mode apply`

</details>

<details>
	<summary>
		4.3 Results will be exported where:
	</summary>
	
<br />

"logs/train" includes a log file and the best-trained model

"logs/app" includes a log file and laz files with a post-regression attribute (as the extra-byte scalar field "AttrPred").

</details>

You can apply the trained model to a much larger area, e.g., the 3D tree-wise biomass map at the landscape level, converted based on the tree DBH and height as below:

<p align="center">
    <img src='https://github.com/truebelief/artemis_treescaling/assets/8785889/64c472b4-6327-4f86-813f-cc4efee883cb' width=600>
</p>

*There's a noticeable gap in the application of cutting-edge AI techniques, which often emphasizes creating impressive 'toy' projects and engaging in superficial competitions for accuracy ranking. This focus tends to prioritize visual appeal and immediate dopamine gratification rather than addressing more substantial challenges. These include the intricate tasks of managing complex environmental data, gaining deeper insights into the physical and spiritual realms, and tackling significant societal issues.


