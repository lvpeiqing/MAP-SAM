# MAP-SAM: Mutual Distillation between CNN and SAM with Adaptive Prototype Prompting for Few-Shot Medical Image Segmentation

![Model Framework](./MAPSAM.png)
(a) Overall architecture of the proposed MAP-SAM, which employs a dual-branch network comprising a CNN and SAM. Given a support image $I^s$ and a query image $I^q$, the CNN branch first utilizes the Adaptive Cross-Level Prototype Generator (ACPG) to extract cross-level global and local prototypes. These prototypes, together with the query features, are then fed into the Multi-Granularity Prototype Fusion Module (MGPFM) to generate robust CNN-based query predictions. Next, an Automated Prompt Generation Module comprising the Adaptive Dual-Boundary Prompt Generation (ADBPG) (b) and Uncertainty-Aware Prompt Generation (UAPG) (c) modules converts the CNN predictions into sparse and dense prompts to guide SAM's query predictions. Finally, mutual knowledge distillation is performed between the CNN and SAM outputs.

### Abstract

Few-shot Medical Image Segmentation (FSMIS) aims to accurately segment novel categories using minimal labeled support data, thereby alleviating data scarcity and annotation burden. Existing CNN-based FSMIS methods, however, remain limited by suboptimal prototype representations and insufficient generalization. Although the Segment Anything Model (SAM) exhibits strong zero-shot capability, its application to medical segmentation is constrained by reliance on manual prompts and limited task-specific semantic adaptation. To address these limitations, we propose MAP-SAM, a novel FSMIS framework integrating adaptive prototype prompting and CNN–SAM mutual distillation. To preserve low-level structural details often suppressed by conventional masked average pooling, we introduce an adaptive cross-level prototype generator (ACPG), which dynamically fuses shallow spatial details with deep semantic features to construct discriminative global and local prototypes across hierarchical layers. A multi-granularity prototype fusion module (MGPFM) further performs cascaded learnable aggregation of query predictions from prototypes at different semantic levels and spatial granularities, yielding robust CNN-based segmentation. To remove the need for manual SAM prompts, we design a CNN-driven automatic prompt generation module that combines adaptive dual-boundary prompt generation (ADBPG) and uncertainty-aware prompt generation (UAPG) to produce sparse and dense prompts, respectively, guiding SAM to delineate ambiguous boundaries. Finally, a mutual knowledge distillation strategy establishes a closed-loop learning paradigm that enables complementary exchange between task-specific semantic knowledge and general boundary priors. Experiments on three public datasets demonstrate that MAP-SAM achieves state-of-the-art FSMIS performance. 

### Dependencies

Please install following essential dependencies:

```
dcm2niix
json5==0.8.5
jupyter==1.0.0
nibabel==2.5.1
numpy==1.21.6
opencv-python==4.1.0.25
Pillow>=8.1.1
sacred==0.7.5
scikit-image==0.19.3
SimpleITK==1.2.3
torch==1.13.1+cu116
torchvision==0.14.1+cu116
tqdm==4.62.3
```

### Data sets and pre-processing

Download:

1. [Combined Healthy Abdominal Organ Segmentation data set](https://chaos.grand-challenge.org/)
2. [Multi-sequence Cardiac MRI Segmentation data set](https://zmiclab.github.io/projects/mscmrseg19/) (bSSFP fold)
3. [Multi-Atlas Abdomen Labeling Challenge](https://www.synapse.org/#!Synapse:syn3193805/wiki/218292)

Pre-processing is performed according to [Ouyang et al.](https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation/tree/2f2a22b74890cb9ad5e56ac234ea02b9f1c7a535) and we follow the procedure on their github repository.

Preprocessed data and supervoxels can be downloaded via [RPT](https://github.com/YazhouZhu19/RPT) .

### Training

1. Compile `./supervoxels/felzenszwalb_3d_cy.pyx` with cython (`python ./supervoxels/setup.py build_ext --inplace`) and run `./supervoxels/generate_supervoxels.py` 
2. Download pre-trained ResNet-101 weights [vanilla version](https://download.pytorch.org/models/resnet101-63fe2227.pth) or [deeplabv3 version](https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth) and put them in the checkpoint folder, then replace the absolute path in the code `./models/encoder.py`. 
3. Download the pretrained SAM-b weights [vit_b(default)](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) and put them in the checkpoint folder, then replace the absolute path in the code `./Model/sam/bulid_sam.py`.
4. Run `./script/train.sh` 

### Inference

Run `./script/test.sh`

### Acknowledgement

Our implementation is based on the works: [SSL-ALPNet](https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation), [ADNet](https://github.com/sha168/ADNet), [QNet](https://github.com/ZJLAB-AMMI/Q-Net) and [HCPMNet](https://github.com/zmcheng9/HCPMNet).
