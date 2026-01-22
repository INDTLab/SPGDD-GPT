
# SPGDD-GPT: Image-Text-Driven Generic Defect Diagnosis Using a Self-prompted Large Vision-Language Model




Shengwang An, Xinghui Dong



****




<span id='environment'/>

### 1. Running SPGDD-GPT Demo <a href='#all_catelogue'>[Back to Top]</a>

<span id='install_environment'/>

#### 1.1 Environment Installation

Clone the repository locally:

```
git clone https://github.com/INDTLab/SPGDD-GPT.git
```

Install the required packages:

```
pip install -r requirements.txt
```

<span id='download_imagebind_model'/>

#### 1.1 Prepare ImageBind Checkpoint:

You can download the pre-trained ImageBind model using [this link](https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth). After downloading, put the downloaded file (imagebind_huge.pth) in [[./pretrained_ckpt/imagebind_ckpt/]](./pretrained_ckpt/imagebind_ckpt/) directory. 

<span id='download_vicuna_model'/>

#### 1.2 Prepare Vicuna Checkpoint:

To prepare the pre-trained Vicuna model, please follow the instructions provided [[here]](./pretrained_ckpt#1-prepare-vicuna-checkpoint).

<span id='download_spgdd-gpt'/>

#### 1.4 Prepare Delta Weights of SPGDD-GPT:

We use the pre-trained parameters from [PandaGPT](https://github.com/yxuansu/PandaGPT) to initialize our model. You can get the weights of PandaGPT trained with different strategies in the table below. In our experiments and online demo, we use the Vicuna-7B and `openllmplayground/pandagpt_7b_max_len_1014` due to the limitation of computation resource. Better results are expected if switching to Vicuna-12B.

Please put the downloaded 7B/12B delta weights file (pytorch_model.pt) in the [./pretrained_ckpt/pandagpt_ckpt/7b/](./pretrained_ckpt/pandagpt_ckpt/7b/) or [./pretrained_ckpt/pandagpt_ckpt/12b/](./pretrained_ckpt/pandagpt_ckpt/12b/) directory. 

<span id='running_demo'/>

#### 1.5. Deploying Demo

Upon completion of previous steps, you can run the demo locally as
```bash
python web_demo.py
```

****

<span id='train_spgdd-gpt'/>

### 2. Train Your Own SPGDD-GPT  <a href='#all_catelogue'>[Back to Top]</a>

**Prerequisites:** Before training the model, making sure the environment is properly installed and the checkpoints of ImageBind, Vicuna and PandaGPT are downloaded. 

<span id='data_preparation'/>

#### 2.1 Data Preparation:

You can download MVTec-AD dataset from [[this link]](https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads) and VisA from [[this link]](https://github.com/amazon-science/spot-diff). You can also download pre-training data of PandaGPT from [[here]](https://huggingface.co/datasets/openllmplayground/pandagpt_visual_instruction_dataset/tree/main). After downloading, put the data in the [[./data]](./data/) directory.

The directory of [[./data]](./data/) should look like:

```
data
|---TADD_text.json
|---TADD
|-----|-- mvtec_anomaly_detection
|-----|-----|----- bottle
|-----|-----|----- capsule
|-----|-- VisA
|-----|-- ...
```



<span id='training_configurations'/>

#### 2.1 Training Configurations

The table below show the training hyperparameters used in our experiments. The hyperparameters are selected based on the constrain of our computational resources, i.e. 2 x RTX4090 GPUs.

| **Base Language Model** | **Epoch Number** | **Batch Size** | **Learning Rate** | **Maximum Length** |
| :---------------------: | :--------------: | :------------: | :---------------: | :----------------: |
|        Vicuna-7B        |        50        |       16       |       1e-2        |        1014        |



<span id='model_training'/>

#### 2.2 Training SPGDD-GPT

To train SPGDD-GPT on MVTec-AD dataset, please run the following commands:
```yaml
cd ./code
bash ./scripts/train_mvtec.sh
```

The key arguments of the training script are as follows:
* `--data_path`: The data path for the json file `TADD_text.json`.
* `--image_root_path`: The root path for training images of PandaGPT.
* `--imagebind_ckpt_path`: The path of ImageBind checkpoint.
* `--vicuna_ckpt_path`: The directory that saves the pre-trained Vicuna checkpoints.
* `--max_tgt_len`: The maximum sequence length of training instances.
* `--save_path`: The directory which saves the trained delta weights. This directory will be automatically created.
* `--log_path`: The directory which saves the log. This directory will be automatically created.

Note that the epoch number can be set in the `epochs` argument at [./code/config/openllama_peft.yaml](code/config/openllama_peft.yaml) file and the learning rate can be set in  [./code/dsconfig/openllama_peft_stage_1.json](code/dsconfig/openllama_peft_stage_1.json)









 