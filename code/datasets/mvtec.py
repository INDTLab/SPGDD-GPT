import os
from torch.utils.data import Dataset
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


from .self_sup_tasks import patch_ex

WIDTH_BOUNDS_PCT = {'DeepCrack':((0.03, 0.4), (0.03, 0.4)),'KolektorSDD':((0.03, 0.4), (0.03, 0.4)),
                    'KolektorSDD2':((0.03, 0.4), (0.03, 0.4)),'NEU':((0.03, 0.4), (0.03, 0.4)),
                    'OUCCrack':((0.05, 0.4), (0.05, 0.4)), 'Volker':((0.03, 0.15), (0.03, 0.4)),
                    'Rissbilder':((0.03, 0.35), (0.03, 0.35)), 'RSDD':((0.03, 0.4), (0.03, 0.4)),
                    'RSDD2':((0.03, 0.2), (0.03, 0.4)),'CFD':((0.03, 0.12), (0.03, 0.12)),
                    'Crack500':((0.03, 0.4), (0.03, 0.2)), 'CrackTree200':((0.03, 0.4), (0.03, 0.4)),
                    'Aitex':((0.03, 0.4), (0.03, 0.2)),'Eugen_Miller':((0.03, 0.4), (0.03, 0.4)), 'MT':((0.03, 0.4), (0.03, 0.4)),
                    'GAPs':((0.03, 0.4), (0.03, 0.4))}


#NUM_PATCHES 字典指定了每种物体类别在图像分割时应生成的补丁数量
#根据物体的复杂性和所需的数据增强程度来设置补丁数量。简单的物体可能需要较少的补丁，而复杂的物体或需要多尺度分析的物体可能需要更多的补丁。

NUM_PATCHES = {'DeepCrack':3,'KolektorSDD':3,'KolektorSDD2':3 ,'NEU':3, 'OUCCrack':3, 'Volker':3, 'Rissbilder':3, 'RSDD':3,
               'RSDD2':3, 'CFD':4, 'Crack500':3, 'CrackTree200':3, 'Aitex':4,
               'Eugen_Miller':4, 'MT':4, 'GAPs':4}


#INTENSITY_LOGISTIC_PARAMS 字典为每个物体类别提供了逻辑斯蒂回归模型的参数，这些参数可能用于调整图像的对比度或亮度，或者用于图像的归一化。
#设置方法：这些参数通常基于图像的亮度分布来设置。例如，如果图像的亮度变化不大，可能需要较小的参数值来保持细节；如果图像的亮度变化范围很广，则可能需要较#大的参数值来压缩动态范围。

# k, x0 pairs
INTENSITY_LOGISTIC_PARAMS = {'DeepCrack':(1/3, 7),'KolektorSDD':(1/3, 7), 'KolektorSDD2':(1/3, 7), 'NEU':(1/12, 24), 'OUCCrack':(1/12, 24),
                             'Volker':(1/2, 4), 'Rissbilder':(1/12, 24), 'RSDD':(1/3, 7),
            'RSDD2':(1/3, 7), 'CFD':(1, 3), 'Crack500':(1/6, 15), 'CrackTree200':(1/6, 15), 'Aitex':(1/6, 15),
            'Eugen_Miller':(1/3, 7), 'MT':(1/3, 7), 'GAPs':(1/3, 7)}

# bottle is aligned but it's symmetric under rotation
#这个列表包含了那些没有对齐或者在旋转下对称的物体类别名称。这可能用于指示在图像处理时需要特别注意这些物体，因为它们可能需要额外的对齐或旋转不变性处理。
UNALIGNED_OBJECTS = []

#BACKGROUND 字典为每个物体类别提供了背景亮度和阈值参数，这些参数用于区分图像中的前景物体和背景。
#设置方法：
#亮度（brightness）：根据图像的平均亮度来设置。如果背景较亮，则需要较高的亮度值。
#阈值（threshold）：用于确定像素是否属于背景。这个值通常通过实验来确定，以确保能够有效地分离前景和背景。

# brightness, threshold pairs
BACKGROUND = {'DeepCrack':(200, 60), 'KolektorSDD':(200, 60), 'RSDD2':(200, 60),
              'OUCCrack':(20, 20), 'KolektorSDD2':(20, 20), 'RSDD':(20, 20)}

#OBJECTS 列表中的物体可能是三维的，而 TEXTURES 列表中的纹理可能是二维的表面。这些分类有助于在数据加载和处理时区分不同类型的图像。
OBJECTS = ['DeepCrack','KolektorSDD', 'KolektorSDD2','NEU']
TEXTURES = ['OUCCrack','Volker', 'Rissbilder', 'RSDD', 'RSDD2','CFD', 'Crack500', 'CrackTree200', 'Aitex', 'Eugen_Miller',  'MT',
               'GAPs']

describles = {}
describles[
    'CFD'] = "This is a photo of a CFD for anomaly detection, which should be round, without any damage, flaw, defect, scratch, hole or broken part."
describles[
    'Crack500'] = "This is a photo of Crack500 for anomaly detection, cables cannot be missed or swapped, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles[
    'CrackTree200'] = "This is a photo of a CrackTree200 for anomaly detection, which should be black and orange, with print '500', without any damage, flaw, defect, scratch, hole or broken part."
describles[
    'DeepCrack'] = "This is a photo of DeepCrack for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles[
    'Eugen_Miller'] = "This is a photo of Eugen_Miller for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles[
    'GAPs'] = "This is a photo of a GAPs for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles[
    'KolektorSDD'] = "This is a photo of KolektorSDD for anomaly detection, which should be brown and without any damage, flaw, defect, scratch, hole or broken part."
describles[
    'KolektorSDD2'] = "This is a photo of a KolektorSDD2 for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part, and shouldn't be fliped."
describles[
    'MT'] = "This is a photo of a MT for anomaly detection, which should be white, with print 'FF' and red patterns, without any damage, flaw, defect, scratch, hole or broken part."
describles[
    'NEU'] = "This is a photo of a NEU for anomaly detection, which tail should be sharp, and without any damage, flaw, defect, scratch, hole or broken part."
describles[
    'OUCCrack'] = "This is a photo of OUCCrack for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles[
    'Rissbilder'] = "This is a photo of a Rissbilder for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles[
    'RSDD'] = "This is a photo of a RSDD for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."
describles[
    'RSDD2'] = "This is a photo of RSDD2 for anomaly detection, which should be brown with patterns, without any damage, flaw, defect, scratch, hole or broken part."
describles[
    'Volker'] = "This is a photo of a Volker for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."

describles[
    'Aitex'] = "This is a photo of a Aitex for anomaly detection, which should be without any damage, flaw, defect, scratch, hole or broken part."

class MVtecDataset(Dataset):
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        # self.transform = transform
        self.transform = transforms.Resize(
                                (240, 240), interpolation=transforms.InterpolationMode.BICUBIC
                            )
        
        self.norm_transform = transforms.Compose(
                            [
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    mean=(0.48145466, 0.4578275, 0.40821073),
                                    std=(0.26862954, 0.26130258, 0.27577711),
                                ),
                            ]
                        )

        self.paths = []
        self.x = []
        for root, dirs, files in os.walk('/data2/shengwang/AnomalyGPT/data/allDataOfMy/'):
            for file in files:
                file_path = os.path.join(root, file)
                class_name = file_path.split('/')[-4]
                if "train" in file_path and "good" in file_path and 'png' in file:
                    self.paths.append(file_path)
                    self.x.append(self.transform(Image.open(file_path).convert('RGB')))


        self.prev_idx = np.random.randint(len(self.paths))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):

        img_path, x = self.paths[index], self.x[index]
        class_name = img_path.split('/')[-4]

        self_sup_args={'width_bounds_pct': WIDTH_BOUNDS_PCT.get(class_name),
                    'intensity_logistic_params': INTENSITY_LOGISTIC_PARAMS.get(class_name),
                    'num_patches': 3, #if single_patch else NUM_PATCHES.get(class_name),
                    'min_object_pct': 0,
                    'min_overlap_pct': 0.25,
                    'gamma_params':(2, 0.05, 0.03), 'resize':True, 
                    'shift':True, 
                    'same':False, 
                    'mode':cv2.NORMAL_CLONE,
                    'label_mode':'logistic-intensity',
                    'skip_background': BACKGROUND.get(class_name)}
        if class_name in TEXTURES:
            self_sup_args['resize_bounds'] = (.5, 2)

        # img_path.split('/')[-1].split('.')[0] + '_mask.' + img_path.split('/')[-1].split('.')[1]     #159.png->159.mask.png
        if((class_name in ['capsule','screw']) and ('good' not in img_path)):
            mask_path = img_path.replace('good', 'ground_truth')
            mask_path.split('/')[-1] = mask_path.split('/')[-1].split('.')[0] + '_mask.' + mask_path.split('/')[-1].split('.')[1]
            x = np.asarray(x)
            origin = x
            p = self.x[self.prev_idx]
            if self.transform is not None:
                p = self.transform(p)
            p = np.asarray(p)
            mask = torch.tensor(np.asarray(self.transform(Image.open(mask_path).convert('L')))).unsqueeze(-1).float()
            self.prev_idx = index
            origin = self.norm_transform(origin)
            x = self.norm_transform(x)
            centers = [(200,200)]
        else:
            x = np.asarray(x)
            origin = x

            p = self.x[self.prev_idx]
            if self.transform is not None:
                p = self.transform(p)
            p = np.asarray(p)
            x, mask, centers = patch_ex(x, p, **self_sup_args)
            mask = torch.tensor(mask[None, ..., 0]).float()
            self.prev_idx = index

            origin = self.norm_transform(origin)
            x = self.norm_transform(x)

        if len(centers) > 0:
            position = []
            for center in centers:
                center_x = center[0] / 240
                center_y = center[1] / 240

                if center_x <= 1/3 and center_y <= 1/3:
                    position.append('top left')
                elif center_x <= 1/3 and center_y > 1/3 and center_y <= 2/3:
                    position.append('top')
                elif center_x <= 1/3 and center_y > 2/3:
                    position.append('top right')

                elif center_x <= 2/3 and center_y <= 1/3:
                    position.append('left')
                elif center_x <= 2/3 and center_y > 1/3 and center_y <= 2/3:
                    position.append('center')
                elif center_x <= 2/3 and center_y > 2/3:
                    position.append('right')

                elif center_y <= 1/3:
                    position.append('bottom left')
                elif center_y > 1/3 and center_y <= 2/3:
                    position.append('bottom')
                elif center_y > 2/3:
                    position.append('bottom right')

            conversation_normal = []
            conversation_normal.append({"from":"human","value": describles[class_name] + " Is there any anomaly in the image?"})
            conversation_normal.append({"from":"gpt","value":"No, there is no anomaly in the image."})



            conversation_abnormal = []
            conversation_abnormal.append({"from":"human","value": describles[class_name] + " Is there any anomaly in the image?"})



            if len(centers) > 1:
                abnormal_describe =  "Yes, there are " + str(len(centers)) + " anomalies in the image, they are at the "
                for i in range(len(centers)):
                    if i == 0:
                        abnormal_describe += position[i]

                    elif i == 1 and position[i] != position[i-1]:
                        if i != len(centers) - 1:
                            abnormal_describe += ", "
                            abnormal_describe += position[i]
                        else:
                            abnormal_describe += " and " + position[i] + " of the image."

                    elif i == 1 and position[i] == position[i-1]:
                        if i == len(centers) - 1:
                            abnormal_describe += " of the image."

            else:
                abnormal_describe = "Yes, there is an anomaly in the image, at the " + position[0] + " of the image."

            conversation_abnormal.append({"from":"gpt","value":abnormal_describe})

        else:
            print("no mask")
            conversation_normal = []
            conversation_normal.append({"from":"human","value":describles[class_name] + " Is there any anomaly in the image?"})
            conversation_normal.append({"from":"gpt","value":"No, there is no anomaly in the image."})

            conversation_abnormal = conversation_normal

        return origin, conversation_normal, x, conversation_abnormal, class_name, mask, img_path



    def collate(self, instances):

        images = []
        texts = []
        class_names = []
        masks = []
        img_paths = []
        for instance in instances:
            images.append(instance[0])
            texts.append(instance[1])
            class_names.append(instance[4])
            masks.append(torch.zeros_like(torch.Tensor(instance[5]).float()))
            img_paths.append(instance[6])

            images.append(instance[2])
            texts.append(instance[3])
            class_names.append(instance[4])
            masks.append(instance[5])
            img_paths.append(instance[6])

        return dict(
            images=images,
            texts=texts,
            class_names=class_names,
            masks=masks,
            img_paths=img_paths
        )