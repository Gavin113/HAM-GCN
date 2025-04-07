# Hand-Aware Masked Graph Convolutional Network for Sign Language Recognition
This repository includes Python (PyTorch) source code of the paper.

![](./figures/main_model.png)
![](./figures/HAM-GC-Block.png)

## Requirements

```bash
python==3.10.9
torch==1.12.1
tensorboardX==2.6.2.2
scikit-learn==1.2.1
tqdm==4.64.1
numpy==1.23.5
PyYAML==6.0.1
natsort==8.3.1
```
Run ```pip install -e torchlight``` to install torchlight.
## Datasets
* Download the original datasets, including [NMFs_CSL](https://ustc-slr.github.io/datasets/), [MSASL](https://www.microsoft.com/en-us/research/project/ms-asl/) and [AUTSL](https://cvml.ankara.edu.tr/datasets/) from the official websites.

* Use the [data_gen/gen_poses.py](data_gen/gen_poses.py) with the pretrained model of [Hrnet on Coco-Wholebody](https://drive.google.com/file/d/1f_c3uKTDQ4DR3CrwMSI8qdsTKJvKVt7p/view) to extract the 2D keypoints for sign language videos.

* Use the [data_gen/gen_masks.py](data_gen/gen_masks.py) to generate the temporal mask for the keypoints.

* The final data is formatted as follows:
```
    Data
    ├── AUTSL
    ├── MSASL
    └── NMFs_CSL
        ├── nmf_train.npz
        ├── nmf_train_mask_T.npz
        ├── nmf_test.npz
        └── nmf_test_mask_T.npz
```

## Train the Model
```bash
python main_ham_gcn.py --config config/NMFs_CSL/nmfs_csl_joint_gcn_mask.yaml --phase train --device 0
```
## Test the Model
```bash
python main_ham_gcn.py --config config/NMFs_CSL/nmfs_csl_joint_gcn_mask.yaml --phase test --weights work_dir/nmfs/joint_gcn_mask/best.pt --device 0
```

