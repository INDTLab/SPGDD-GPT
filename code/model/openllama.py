from header import *
import torch.nn.functional as F
import torch.nn as nn
from .ImageBind import *
from .ImageBind import data
from .modeling_llama import LlamaForCausalLM
from .AnomalyGPT_models import LinearLayer, PromptLearner1
from transformers import StoppingCriteria, StoppingCriteriaList
from utils.loss import FocalLoss, BinaryDiceLoss
import kornia as K
from peft import LoraConfig, TaskType, get_peft_model
import torch
from torch.nn.utils import rnn
from transformers import LlamaTokenizer, LlamaForCausalLM, LlamaConfig
from PromptAD import *

from PromptAD import TripletLoss

CLASS_NAMES = ['OUCCrack','Volker', 'Rissbilder', 'RSDD', 'RSDD2','CFD', 'Crack500', 'CrackTree200', 'Aitex','DeepCrack', 'Eugen_Miller', 'KolektorSDD', 'KolektorSDD2', 'MT',
               'NEU','GAPs' ,'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper', 'object',
               'candle', 'cashew', 'chewinggum', 'fryum', 'macaroni', 'pcb', 'pipe fryum']

prompt_normal = ['{}', 'flawless {}', 'perfect {}', 'unblemished {}', '{} without flaw', '{} without defect', '{} without damage']
prompt_abnormal = ['damaged {}', 'broken {}', '{} with flaw', '{} with defect', '{} with damage']

prompt_state = [prompt_normal, prompt_abnormal]
prompt_templates = ['a photo of a {}.', 'a photo of the {}.']
# prompt_templates = [
#                         'a cropped photo of the {}.', 'a cropped photo of a {}.', 'a close-up photo of a {}.', 'a close-up photo of the {}.',
#                         'a bright photo of the {}.', 'a bright photo of a {}.', 'a dark photo of a {}.', 'a dark photo of the {}.',
#                         'a dark photo of the {}.', 'a dark photo of a {}.', 'a jpeg corrupted photo of a {}.', 'a jpeg corrupted photo of the {}.',
#                         'a blurry photo of the {}.', 'a blurry photo of a {}.', 'a photo of a {}.', 'a photo of the {}.',
#                         'a photo of the small {}.', 'a photo of a small {}.', 'a photo of the large {}.', 'a photo of a large {}.',
#                         'a photo of the {} for visual insprction.', 'a photo of a {} for visual insprction.',
#                         'a photo of the {} for anomaly detection.', 'a photo of a {} for anomaly detection.'
#                         ]
objs = ['OUCCrack','Volker', 'Rissbilder', 'RSDD', 'RSDD2','CFD', 'Crack500', 'CrackTree200', 'Aitex','DeepCrack', 'Eugen_Miller', 'KolektorSDD', 'KolektorSDD2', 'MT',
               'NEU','GAPs' ,'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper', 'object',
        'candle', 'cashew', 'chewinggum', 'fryum', 'macaroni', 'pcb', 'pipe fryum', 'macaroni1', 'macaroni2','pcb1', 'pcb2', 'pcb3', 'pcb4', 'capsules']

prompt_sentences = {}

#for obj in objs:
#    prompt_sentence_obj = []
#    for i in range(len(prompt_state)):
#        prompted_state = [state.format(obj) for state in prompt_state[i]]
#        prompted_sentence = []
#        for s in prompted_state:
#            for template in prompt_templates:
#                prompted_sentence.append(template.format(s))
#        prompted_sentence = data.load_and_transform_text(prompted_sentence, torch.cuda.current_device())
#        prompt_sentence_obj.append(prompted_sentence)
#    prompt_sentences[obj] = prompt_sentence_obj



def encode_text_with_prompt_ensemble(model, obj, device):

    global prompt_sentences
    normal_sentences = []
    abnormal_sentences = []
    for idx in range(len(obj)):
        sentence = prompt_sentences[obj[idx].replace('_', ' ')]
        normal_sentences.append(sentence[0])
        abnormal_sentences.append(sentence[1])

    normal_sentences = torch.cat(normal_sentences).to(device)
    abnormal_sentences = torch.cat(abnormal_sentences).to(device)

    class_embeddings_normal = model({ModalityType.TEXT: normal_sentences})[ModalityType.TEXT][0]
    class_embeddings_abnormal = model({ModalityType.TEXT: abnormal_sentences})[ModalityType.TEXT][0]
    # class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)

    class_embeddings_normal = class_embeddings_normal.reshape((len(obj), len(prompt_templates) * len(prompt_normal), 1024))
    class_embeddings_normal = class_embeddings_normal.mean(dim=1, keepdim=True)
    class_embeddings_normal = class_embeddings_normal / class_embeddings_normal.norm(dim=-1, keepdim=True)

    class_embeddings_abnormal = class_embeddings_abnormal.reshape((len(obj), len(prompt_templates) * len(prompt_abnormal), 1024))
    class_embeddings_abnormal = class_embeddings_abnormal.mean(dim=1, keepdim=True)
    class_embeddings_abnormal = class_embeddings_abnormal / class_embeddings_abnormal.norm(dim=-1, keepdim=True)

    text_features = torch.cat([class_embeddings_normal, class_embeddings_abnormal], dim=1)

    return text_features



class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops = [], encounters=1):
        super().__init__()
        self.stops = stops
        self.ENCOUNTERS = encounters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        stop_count = 0
        for stop in self.stops:
            stop_count = (stop == input_ids[0]).sum().item()
        if stop_count >= self.ENCOUNTERS:
            return True
        return False

def build_one_instance(tokenizer, conversation):
    text_list = []

    turn_num = len(conversation)

    print(turn_num)
    input_ids, target_ids = [], []
    for i in range(turn_num):
        turn = conversation[i]
        role = turn['from']
        if i == 0: # the first human turn
            assert role == 'human'
            text = turn['value'] + '\n### Assistant:'
            one_input_id = tokenizer(text, add_special_tokens=False).input_ids
            input_ids += one_input_id
            target_ids += [-100]*len(one_input_id) # do not perform loss regression on human prompt
        else:
            if role == 'human':
                text = 'Human: ' + turn['value'] + '\n### Assistant:'
                one_input_id = tokenizer(text, add_special_tokens=False).input_ids
                input_ids += one_input_id
                target_ids += [-100]*len(one_input_id)
            elif role == 'gpt':
                text = turn['value'] + '\n###'
                one_input_id = tokenizer(text, add_special_tokens=False).input_ids
                input_ids += one_input_id
                target_ids += one_input_id
            else:
                raise Exception('Wrong Role!!!')
        text_list.append(text)
        assert len(input_ids) == len(target_ids)
    return text_list, input_ids, target_ids

def process_batch_instance(tokenizer, batch_of_conversations, max_tgt_len):
    batch_input_ids, batch_target_ids = [], []
    if isinstance(batch_of_conversations, list):
        # 如果最外层只有一个列表，则再添加一个
        if not isinstance(batch_of_conversations[0], list):
            batch_of_conversations = [batch_of_conversations]
        else:
            batch_of_conversations = batch_of_conversations
    print(batch_of_conversations)
    for conversation in batch_of_conversations:
        print(conversation)
        _, one_input_ids, one_target_ids = build_one_instance(tokenizer, conversation)
        batch_input_ids.append(torch.LongTensor(one_input_ids))
        batch_target_ids.append(torch.LongTensor(one_target_ids))
    input_ids = rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    target_ids = rnn.pad_sequence(batch_target_ids, batch_first=True, padding_value=-100)
    assert input_ids.size() == target_ids.size()
    input_ids = input_ids[:,:max_tgt_len]
    target_ids = target_ids[:,:max_tgt_len]
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    assert attention_mask.size() == input_ids.size()
    return input_ids, target_ids, attention_mask.long()

def find_first_file_in_directory(directory_path):
    try:
        file_list = os.listdir(directory_path)
        for item in file_list:
            item_path = os.path.join(directory_path, item)
            if os.path.isfile(item_path):
                return item_path
        return None

    except OSError as e:
        print(f"Error while accessing directory: {e}")
        return None


PROMPT_START = '### Human: <Img>'
class OpenLLAMAPEFTModel(nn.Module):

    '''LoRA for LLaMa model'''

    def __init__(self, **args):
        super(OpenLLAMAPEFTModel, self).__init__()
        self.args = args
        imagebind_ckpt_path = args['imagebind_ckpt_path']
        vicuna_ckpt_path = args['vicuna_ckpt_path']
        max_tgt_len = args['max_tgt_len']
        stage = args['stage']

        print (f'Initializing visual encoder from {imagebind_ckpt_path} ...')

        #self.visual_encoder, self.visual_hidden_size = imagebind_model.imagebind_huge(args)
        #imagebind_ckpt = torch.load(imagebind_ckpt_path, map_location=torch.device('cpu'))
        #self.visual_encoder.load_state_dict(imagebind_ckpt, strict=True)

        self.iter = 0

        self.image_decoder = LinearLayer(1280, 1024, 4)

        self.prompt_learner1 = PromptLearner1(1, 4096)

        self.loss_focal = FocalLoss()
        self.loss_dice = BinaryDiceLoss()
        self.weights_cos_nolAndAbnol =nn.Parameter(torch.ones(896).view(1, 1, 1, -1)) 


        # free vision encoder
        #for name, param in self.visual_encoder.named_parameters():
            #param.requires_grad = False
        #self.visual_encoder.eval()
        #print ('Visual encoder initialized.')

        print (f'Initializing language decoder from {vicuna_ckpt_path} ...')
        
        # add the lora module
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=self.args['lora_r'], 
            lora_alpha=self.args['lora_alpha'], 
            lora_dropout=self.args['lora_dropout'],
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']
        )

        self.llama_model = LlamaForCausalLM.from_pretrained(vicuna_ckpt_path)
        self.llama_model = get_peft_model(self.llama_model, peft_config)
        self.llama_model.print_trainable_parameters()

        self.llama_tokenizer = LlamaTokenizer.from_pretrained(vicuna_ckpt_path, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        self.llama_tokenizer.padding_side = "right"
        print ('Language decoder initialized.')

        #self.llama_proj = nn.Linear(
        #    self.visual_hidden_size, self.llama_model.config.hidden_size
        #)
        self.myModel_llama_proj = nn.Linear(
            640, self.llama_model.config.hidden_size
        )

        self.max_tgt_len = max_tgt_len
        self.device = torch.cuda.current_device()
        self.criterion_myModel = nn.CrossEntropyLoss().to(self.device)
        self.criterion_tip_myModel = TripletLoss(margin=0.0)
        
        
        kwargs = {'dataset': 'mvtec', 'class_name': 'carpet', 'img_resize': 240, 'img_cropsize': 240, 'resolution': 400,
                  'batch_size': 800, 'vis': True, 'root_dir': './result', 'load_memory': True, 'cal_pro': False,
                  'seed': 111, 'gpu_id': 0, 'pure_test': False, 'k_shot': 1, 'backbone': 'ViT-B-16-plus-240',
                  'pretrained_dataset': 'laion400m_e32', 'version': '',
                  'use_cpu': 0, 'n_ctx': 4, 'n_ctx_ab': 1, 'n_pro': 1, 'n_pro_ab': 4,
                  'Epoch': 100, 'lr': 0.002, 'momentum': 0.9, 'weight_decay': 0.0005, 'lambda1': 0.001,
                  'device': 'cuda:0', 'out_size_h': 400, 'out_size_w': 400}

        self.myModel = PromptAD(**kwargs)
        self.myModel = self.myModel.to(self.device)


    def rot90_img(self,x,k):
        # k is 0,1,2,3
        degreesarr = [0., 90., 180., 270., 360]
        degrees = torch.tensor(degreesarr[k]).to(self.llama_model.dtype).to(self.device)
        x = K.geometry.transform.rotate(x, angle = degrees, padding_mode='reflection')
        return x

    def encode_video(self, video_paths):
        inputs = {ModalityType.VISION: data.load_and_transform_video_data(video_paths, self.device)}
        # convert into visual dtype
        inputs = {key: inputs[key].to(self.llama_model.dtype) for key in inputs}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            video_embeds = embeddings[ModalityType.VISION][0] # bsz x 1024
        inputs_llama = self.llama_proj(video_embeds).unsqueeze(1) # bsz x 1 x llama_size
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(self.device) # bsz x 1
        return inputs_llama, atts_llama

    def encode_audio(self, audio_paths):
        inputs = {ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, self.device)}
        # convert into visual dtype
        inputs = {key: inputs[key].to(self.llama_model.dtype) for key in inputs}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            audio_embeds = embeddings[ModalityType.AUDIO][0] # bsz x 1024
        inputs_llama = self.llama_proj(audio_embeds).unsqueeze(1) # bsz x 1 x llama_size
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(self.device) # bsz x 1
        return inputs_llama, atts_llama

    def encode_thermal(self, thermal_paths):
        inputs = {ModalityType.THERMAL: data.load_and_transform_thermal_data(thermal_paths, self.device)}
        # convert into visual dtype
        inputs = {key: inputs[key].to(self.llama_model.dtype) for key in inputs}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            image_embeds = embeddings['thermal'][0] # bsz x 1024
        inputs_llama = self.llama_proj(image_embeds).unsqueeze(1) # bsz x 1 x llama_size
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(self.device) # bsz x 1
        return inputs_llama, atts_llama

    def encode_image(self, image_paths):
        
        inputs = {ModalityType.VISION: data.load_and_transform_vision_data(image_paths, self.device)}
        # convert into visual dtype
        inputs = {key: inputs[key].to(self.llama_model.dtype) for key in inputs}
        with torch.no_grad():
            noNeed1, noNeed2, feature_map1, feature_map2 = self.myModel.encode_image(inputs['vision'])
            embeddings = self.visual_encoder(inputs)
            image_embeds = embeddings['vision'][0] # bsz x 1024
            patch_features = embeddings['vision'][1] # bsz x h*w x 1280
        patch_tokens = self.image_decoder(patch_features) # bsz x h*w x 1024
        inputs_myModel_llama = self.myModel_llama_proj(noNeed1).unsqueeze(1)
        inputs_llama = self.llama_proj(image_embeds).unsqueeze(1) # bsz x 1 x llama_size
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(self.device) # bsz x 1
        return inputs_llama, atts_llama, patch_tokens,inputs_myModel_llama, noNeed2, feature_map1, feature_map2
    
    def encode_image_for_web_demo(self, image_paths):
        inputs = {ModalityType.VISION: data.load_and_transform_vision_data_for_web_demo(image_paths, self.device)}
        # convert into visual dtype
        inputs = {key: inputs[key].to(self.llama_model.dtype) for key in inputs}
        with torch.no_grad():
            noNeed1, noNeed2, feature_map1, feature_map2 = self.myModel.encode_image(inputs['vision'])
            embeddings = self.visual_encoder(inputs)
            image_embeds = embeddings['vision'][0] # bsz x 1024
            patch_features = embeddings['vision'][1] # bsz x h*w x 1280
        patch_tokens = self.image_decoder(patch_features) # bsz x h*w x 1024

        inputs_myModel_llama = self.myModel_llama_proj(noNeed1.float()).unsqueeze(1)
        inputs_llama = self.llama_proj(image_embeds).unsqueeze(1)  # bsz x 1 x llama_size
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(self.device)  # bsz x 1
        return inputs_llama, atts_llama, patch_tokens,inputs_myModel_llama, noNeed2, feature_map1, feature_map2
    
    def encode_image_for_one_shot(self, image_paths):
        inputs = {ModalityType.VISION: data.load_and_transform_vision_data(image_paths, self.device)}
        # convert into visual dtype
        inputs = {key: inputs[key].to(self.llama_model.dtype) for key in inputs}
        with torch.no_grad():
            noNeed1, noNeed2, feature_map1, feature_map2 = self.myModel.encode_image(inputs['vision'])
            #embeddings = self.visual_encoder(inputs)
            #patch_features = embeddings['vision'][1]  # bsz x h*w x 1280
            #for i in range(len(patch_features)):
                #patch_features[i] = patch_features[i].transpose(0, 1)[:, 1:, :]

        return noNeed1, feature_map1, feature_map2
    
    def encode_image_for_one_shot_from_tensor(self, image_tensors):
        if not isinstance(image_tensors, list):
            image_tensors = [image_tensors]
        inputs = {ModalityType.VISION: torch.stack(image_tensors, dim=0).to(self.device)}
        # convert into visual dtype
        inputs = {key: inputs[key].to(self.llama_model.dtype) for key in inputs}
        with torch.no_grad():
            noNeed1, noNeed2, feature_map1, feature_map2 = self.myModel.encode_image(inputs['vision'])
            #embeddings = self.visual_encoder(inputs)
            #patch_features = embeddings['vision'][1]  # bsz x h*w x 1280
            #for i in range(len(patch_features)):
                #patch_features[i] = patch_features[i].transpose(0, 1)[:, 1:, :]

        return feature_map2, feature_map1, feature_map2
    
    def encode_image_for_one_shot_with_aug(self, image_paths):
        image_tensors = data.load_and_transform_vision_data(image_paths, self.device).to(self.llama_model.dtype)
        B,C,H,W = image_tensors.shape
        # print(B,C,H,W)

        rotated_images = torch.zeros((4, B, C, H, W)).to(self.llama_model.dtype).to(self.device)


        for j, degree in enumerate([0, 1, 2, 3]):
            rotated_img = self.rot90_img(image_tensors, degree)
            # 存储旋转后的图像
            rotated_images[j] = rotated_img

        image_tensors = rotated_images.transpose(0,1).reshape(B * 4, C, H, W)

        inputs = {ModalityType.VISION: image_tensors}
        # convert into visual dtype
        inputs = {key: inputs[key] for key in inputs}
        with torch.no_grad():
            noNeed1, noNeed2, feature_map1, feature_map2 = self.myModel.encode_image(inputs['vision'])
            #embeddings = self.visual_encoder(inputs)
            #patch_features = embeddings['vision'][1]  # bsz x h*w x 1280
            #for i in range(len(patch_features)):
                #patch_features[i] = patch_features[i].transpose(0, 1)[:, 1:, :].reshape(B, 4, 289, 1280).reshape(B,4 * 289,1280)

        return feature_map1,feature_map1, feature_map2
    
    def encode_image_from_tensor(self, image_tensors):
        if not isinstance(image_tensors, list):
            image_tensors = [image_tensors]
        inputs = {ModalityType.VISION: torch.stack(image_tensors, dim=0).to(self.device)}
        # convert into visual dtype
        inputs = {key: inputs[key].to(self.llama_model.dtype) for key in inputs}
        with torch.no_grad():
            #2,640   2,225,640   2,225,896  2,225,896
            noNeed1, noNeed2, feature_map1, feature_map2 = self.myModel.encode_image(inputs['vision'])
            #embeddings = self.visual_encoder(inputs)
            #image_embeds = embeddings['vision'][0]  # bsz x 1024
            #patch_features = embeddings['vision'][1]  # bsz x h*w x 1024
        #patch_tokens = self.image_decoder(patch_features)
        inputs_myModel_llama = self.myModel_llama_proj(noNeed1).unsqueeze(1)
        #inputs_llama = self.llama_proj(image_embeds).unsqueeze(1)  # bsz x 1 x llama_size
        #atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(self.device)  # bsz x 1
        return inputs_myModel_llama, feature_map1, feature_map1, inputs_myModel_llama,noNeed2, feature_map1, feature_map2
    
    def encode_image_from_tensor_no_patch(self, image_tensors):
        if not isinstance(image_tensors, list):
            image_tensors = [image_tensors]
        inputs = {ModalityType.VISION: torch.stack(image_tensors, dim=0).to(self.device)}
        # convert into visual dtype
        inputs = {key: inputs[key].to(self.llama_model.dtype) for key in inputs}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            image_embeds = embeddings['vision'][0] # bsz x 1024

        inputs_llama = self.llama_proj(image_embeds).unsqueeze(1) # bsz x 1 x llama_size
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(self.device) # bsz x 1
        return inputs_llama, atts_llama



    def prompt_wrap(self, img_embeds, input_ids, target_ids, attention_mask, anomaly_embedding = None):
        '''
            input_ids, target_ids, attention_mask: bsz x s2
        '''
        input_ids = input_ids.to(self.device) # bsz x s2
        target_ids = target_ids.to(self.device) # bsz x s2
        attention_mask = attention_mask.to(self.device) # bsz x s2

        batch_size = img_embeds.shape[0]
        p_before = PROMPT_START
        p_before_tokens = self.llama_tokenizer(p_before, 
            return_tensors="pt", add_special_tokens=False).to(self.device)
        # peft model need deeper call
        p_before_embeds = self.llama_model.model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1) # bsz x s1 x embed_dim

        p_middle = '</Img> '
        p_middle_tokens = self.llama_tokenizer(p_middle, 
            return_tensors="pt", add_special_tokens=False).to(self.device)
        # peft model need deeper call
        p_middle_embeds = self.llama_model.model.model.embed_tokens(p_middle_tokens.input_ids).expand(batch_size, -1, -1) # bsz x s1 x embed_dim


        p_after_embeds = self.llama_model.model.model.embed_tokens(input_ids).expand(batch_size, -1, -1) # bsz x s2 x embed_dim
        bos = torch.ones([batch_size, 1],
                         dtype=p_before_tokens.input_ids.dtype,
                         device=p_before_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id # bsz x 1
        bos_embeds = self.llama_model.model.model.embed_tokens(bos) # bsz x 1 x embed_dim

        

        if anomaly_embedding != None:
            inputs_embeds = torch.cat([bos_embeds, p_before_embeds, img_embeds, p_middle_embeds, anomaly_embedding, p_after_embeds], dim=1) # bsz x (1+s1+1+s2) x embed_dim
            # create targets
            empty_targets = (
                torch.ones([batch_size, 1+p_before_embeds.size()[1]+1+p_middle_embeds.size()[1] + anomaly_embedding.size()[1]], # 1 (bos) + s1 + 1 (image vector)
                        dtype=torch.long).to(self.device).fill_(-100)  
            ) # bsz x (1 + s1 + 1)
            targets = torch.cat([empty_targets, target_ids], dim=1) # bsz x (1 + s1 + 1 + s2)
            assert inputs_embeds.size()[1] == targets.size()[1]

            atts_prefix = torch.ones([batch_size, 1+p_before_embeds.size()[1]+1+p_middle_embeds.size()[1] + anomaly_embedding.size()[1]], dtype=torch.long).to(self.device) # bsz x (1 + s1 +1)
            attention_mask = torch.cat([atts_prefix, attention_mask], dim=1)
            assert attention_mask.size() == targets.size() # bsz x (1 + s1 + 1 + s2)
            return inputs_embeds, targets, attention_mask 
        else:
            inputs_embeds = torch.cat([bos_embeds, p_before_embeds, img_embeds, p_middle_embeds, p_after_embeds], dim=1) # bsz x (1+s1+1+s2) x embed_dim
            # create targets
            empty_targets = (
                torch.ones([batch_size, 1+p_before_embeds.size()[1]+1+p_middle_embeds.size()[1]], # 1 (bos) + s1 + 1 (image vector)
                        dtype=torch.long).to(self.device).fill_(-100)  
            ) # bsz x (1 + s1 + 1)
            targets = torch.cat([empty_targets, target_ids], dim=1) # bsz x (1 + s1 + 1 + s2)
            assert inputs_embeds.size()[1] == targets.size()[1]

            atts_prefix = torch.ones([batch_size, 1+p_before_embeds.size()[1]+1+p_middle_embeds.size()[1]], dtype=torch.long).to(self.device) # bsz x (1 + s1 +1)
            attention_mask = torch.cat([atts_prefix, attention_mask], dim=1)
            assert attention_mask.size() == targets.size() # bsz x (1 + s1 + 1 + s2)
            return inputs_embeds, targets, attention_mask 


    def forward(self, inputs):
        

        if 'masks' in inputs:

            image_paths = inputs['images']
            img_embeds, _, patch_tokens,img_embeds_myModel,noNeed2, feature_map1, feature_map2 = self.encode_image_from_tensor(image_paths)
            class_name = inputs['class_names'][0]

            loss_pixel = 0
            
            #myModel process text=================
            
            
            

            normal_text_prompt, abnormal_text_prompt_handle, abnormal_text_prompt_learned = self.myModel.prompt_learner(class_name)
            normal_text_features = self.myModel.encode_text_embedding(normal_text_prompt, self.myModel.prompt_learner.tokenized_normal_prompts)
            abnormal_text_features_handle = self.myModel.encode_text_embedding(abnormal_text_prompt_handle,
                                                                    self.myModel.prompt_learner.tokenized_abnormal_prompts_handle)
            abnormal_text_features_learned = self.myModel.encode_text_embedding(abnormal_text_prompt_learned,
                                                                         self.myModel.prompt_learner.tokenized_abnormal_prompts_learned)
            abnormal_text_features = torch.cat([abnormal_text_features_handle, abnormal_text_features_learned], dim=0)
            # compute mean
            mean_ad_handle = torch.mean(F.normalize(abnormal_text_features_handle, dim=-1), dim=0)
            mean_ad_learned = torch.mean(F.normalize(abnormal_text_features_learned, dim=-1), dim=0)
            loss_match_abnormal = (mean_ad_handle - mean_ad_learned).norm(dim=0) ** 2.0
            # compute v2t loss and triplet loss
            normal_text_features_ahchor = normal_text_features.mean(dim=0).unsqueeze(0)
            normal_text_features_ahchor = normal_text_features_ahchor / normal_text_features_ahchor.norm(dim=-1,
                                                                                                      keepdim=True)
            abnormal_text_features_ahchor = abnormal_text_features.mean(dim=0).unsqueeze(0)
            abnormal_text_features_ahchor = abnormal_text_features_ahchor / abnormal_text_features_ahchor.norm(dim=-1,
                                                                                                               keepdim=True)
            abnormal_text_features = abnormal_text_features / abnormal_text_features.norm(dim=-1, keepdim=True)
            l_pos = torch.einsum('nic,cj->nij', noNeed2, normal_text_features_ahchor.transpose(0, 1))
            l_neg_v2t = torch.einsum('nic,cj->nij', noNeed2, abnormal_text_features.transpose(0, 1))
            if self.myModel.precision == 'fp16':
                logit_scale = self.myModel.model.logit_scale.half()
            else:
                logit_scale = self.myModel.model.logit_scalef

            logits_v2t = torch.cat([l_pos, l_neg_v2t], dim=-1) * logit_scale
            
            
            

            
            
          # ======将与文本结合的特征图与正常图做特征差异计算
                # if random.randint(0, 1) == 0 and len(inputs['img_paths']) == len(image_paths):
                # 
                #     normal_paths = []
                #     for path in inputs['img_paths']:
                #         if 'all_anomalygpt' not in path and 'visa' in path.lower():
                #             normal_path = path.replace('Anomaly', 'Normal')
                #             normal_path = find_first_file_in_directory("/".join(normal_path.split('/')[:-1]))
                #         else:
                #             normal_path = path.replace('test', 'train')
                #             normal_path = find_first_file_in_directory("/".join(normal_path.split('/')[:-2]) + '/good')
                #         normal_paths.append(normal_path)
                #     normal_patch_tokens, normal_patch_map1, normal_patch_map2 = self.encode_image_for_one_shot(
                #         normal_paths)
                # 
                #     # ap = torch.mean((torch.abs(normal_patch_map1 - feature_map1)+torch.abs(normal_patch_map2 - feature_map2)) / 2.0
                #     ap = torch.mean((torch.abs(feature_map1 - normal_patch_map1) + torch.abs(
                #         feature_map2 - normal_patch_map2)) / 2.0, dim=-1, keepdim=True)
                #     ap_norm = torch.norm(ap, p=2, dim=1, keepdim=True)
                #     ap = ap / ap_norm
                #     logits_v2t = logits_v2t * ap
                #     # logits_v2t = torch.sigmoid(torch.mean((torch.abs(normal_patch_map1 - feature_map1)+torch.abs(normal_patch_map2 - feature_map2)) / 2.0,  dim=-1, keepdim=True))*logits_v2t
                #     del normal_patch_tokens, normal_patch_map1, normal_patch_map2
            
            margin = 1.0
            

            

            
            B, L, C = logits_v2t.shape
            H = int(np.sqrt(L))

            target_v2t = torch.zeros([logits_v2t.shape[0], logits_v2t.shape[1]], dtype=torch.long).to(self.device)
            loss_v2t = self.criterion_myModel(logits_v2t.permute(0, 2, 1), target_v2t)
            trip_loss = self.criterion_tip_myModel(noNeed2, normal_text_features_ahchor, abnormal_text_features_ahchor)
            loss_pixel = loss_v2t + trip_loss + loss_match_abnormal * 0.001
            
            # loss_pixel = loss_v2t + trip_loss + loss_match_abnormal * 0.001

            # anomaly_maps_myModel = []
            # anomaly_map = torch.softmax(feature_map1, dim=1)
            # anomaly_maps.append(anomaly_map)
            
            anomaly_map_myModel = F.interpolate(logits_v2t.permute(0, 2, 1).view(B, -1, H, H),
                                            size=224, mode='bilinear', align_corners=True)
            anomaly_map_myModel = torch.softmax(anomaly_map_myModel, dim=1)
            
            anomaly_map_all_myModel = anomaly_map_myModel[:,1,:,:].unsqueeze(1)
            del anomaly_map_myModel
            


            anomaly_map_prompts_myModel = self.prompt_learner1(anomaly_map_all_myModel)
            
            
            
            
            
            if random.randint(0, 1) == 0 and len(inputs['img_paths']) == len(image_paths):

                normal_paths = []
                for path in inputs['img_paths']:
                    if 'all_anomalygpt' not in path and 'visa' in path.lower():
                        normal_path = path.replace('Anomaly', 'Normal')
                        normal_path = find_first_file_in_directory("/".join(normal_path.split('/')[:-1]))
                    else:
                        normal_path = path.replace('test', 'train')
                        normal_path = find_first_file_in_directory("/".join(normal_path.split('/')[:-2]) + '/good')
                    normal_paths.append(normal_path)


                query_patch_tokens, query_patch_map1, query_patch_map2 = self.encode_image_for_one_shot_from_tensor(image_paths)
                normal_patch_tokens, normal_patch_map1, normal_patch_map2 = self.encode_image_for_one_shot_with_aug(normal_paths)

                

                sims = []
                B = len(image_paths)
                query_patch_map1 = query_patch_map1.view(B, 225, 1, 896)
                query_patch_map2 = query_patch_map2.view(B, 225, 1, 896)
                normal_patch_map1 = normal_patch_map1.view(B, 1, -1, 896)
                normal_patch_map2 = normal_patch_map2.view(B, 1, -1, 896)
                # query_patch_map1 = query_patch_map1 * self.weights_cos_nolAndAbnol
                # query_patch_map2 = query_patch_map2 * self.weights_cos_nolAndAbnol
                # normal_patch_map1 = normal_patch_map1 * self.weights_cos_nolAndAbnol
                # normal_patch_map2 = normal_patch_map2 * self.weights_cos_nolAndAbnol
                #2,225,900
                cosine_similarity_matrix1 = F.cosine_similarity(query_patch_map1, normal_patch_map1,
                                                               dim=-1)
                cosine_similarity_matrix2 = F.cosine_similarity(query_patch_map2, normal_patch_map2,
                                                               dim=-1)
                #2，225
                sim_max1, _ =torch.max(cosine_similarity_matrix1, dim=-1)
                sim_max2, _ = torch.max(cosine_similarity_matrix2, dim=-1)
                sim_myModel = torch.mean(torch.stack([sim_max1,sim_max2], dim=0), dim=0).reshape(B, 1, 15, 15)
                sim_myModel = F.interpolate(sim_myModel, size=224, mode='bilinear', align_corners=True)
                anomaly_map_all_myModel = 1 - sim_myModel
            anomaly_map_prompts_myModel = self.prompt_learner1(anomaly_map_all_myModel)
                
                
            output_texts = inputs['texts']
            input_ids, target_ids, attention_mask = process_batch_instance(self.llama_tokenizer, output_texts,
                                                                           self.max_tgt_len)

            img_embeds_myModel, targets_myModel, attention_mask_myModel = self.prompt_wrap(img_embeds_myModel, input_ids, target_ids, attention_mask,anomaly_map_prompts_myModel)
            
            
            
            outputs = self.llama_model(
                inputs_embeds=img_embeds_myModel,
                attention_mask=attention_mask_myModel,
                return_dict=True,
                labels=targets_myModel,
            )
            
            loss = outputs.loss
            del img_embeds, _, patch_tokens,img_embeds_myModel,noNeed2, feature_map1, feature_map2
            # calculate the token accuarcy
            chosen_tokens = torch.max(outputs.logits, dim=-1)[1][:, 1:-1]  # [B, S-1]
            labels = targets_myModel[:, 2:]
            gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(torch.long)  # [B*S]
            valid_mask = (labels != -100).reshape(-1)
            valid_tokens = gen_acc & valid_mask  # [B*S]
            gen_acc = valid_tokens.sum().item() / valid_mask.sum().item()

            return loss + loss_pixel, gen_acc


            #=============================
              
        
        else:
            class_name = ''
            try:
    # 假设 inputs 是您的输入字典
                class_name = inputs['class_names']
            except KeyError:
    # 在这里编写替代执行的代码
    # 例如，您可以设置一个默认的 class_name
                class_name = "bottle"
            
            image_paths = inputs['image_paths']
            
            img_embeds, _, patch_tokens,img_embeds_myModel,noNeed2, feature_map1, feature_map2 = self.encode_image_from_tensor(image_paths)

            output_texts = inputs['output_texts'][0]
            
            loss_pixel = 0

            
            normal_text_prompt, abnormal_text_prompt_handle, abnormal_text_prompt_learned = self.myModel.prompt_learner(class_name)
            normal_text_features = self.myModel.encode_text_embedding(normal_text_prompt, self.myModel.prompt_learner.tokenized_normal_prompts)
            abnormal_text_features_handle = self.myModel.encode_text_embedding(abnormal_text_prompt_handle,
                                                                    self.myModel.prompt_learner.tokenized_abnormal_prompts_handle)
            abnormal_text_features_learned = self.myModel.encode_text_embedding(abnormal_text_prompt_learned,
                                                                         self.myModel.prompt_learner.tokenized_abnormal_prompts_learned)
            abnormal_text_features = torch.cat([abnormal_text_features_handle, abnormal_text_features_learned], dim=0)
            # compute mean
            mean_ad_handle = torch.mean(F.normalize(abnormal_text_features_handle, dim=-1), dim=0)
            mean_ad_learned = torch.mean(F.normalize(abnormal_text_features_learned, dim=-1), dim=0)
            loss_match_abnormal = (mean_ad_handle - mean_ad_learned).norm(dim=0) ** 2.0
            # compute v2t loss and triplet loss
            normal_text_features_ahchor = normal_text_features.mean(dim=0).unsqueeze(0)
            normal_text_features_ahchor = normal_text_features_ahchor / normal_text_features_ahchor.norm(dim=-1,
                                                                                                         keepdim=True)
            abnormal_text_features_ahchor = abnormal_text_features.mean(dim=0).unsqueeze(0)
            abnormal_text_features_ahchor = abnormal_text_features_ahchor / abnormal_text_features_ahchor.norm(dim=-1,
                                                                                                               keepdim=True)
            abnormal_text_features = abnormal_text_features / abnormal_text_features.norm(dim=-1, keepdim=True)
            l_pos = torch.einsum('nic,cj->nij', noNeed2, normal_text_features_ahchor.transpose(0, 1))
            l_neg_v2t = torch.einsum('nic,cj->nij', noNeed2, abnormal_text_features.transpose(0, 1))
            if self.myModel.precision == 'fp16':
                logit_scale = self.myModel.model.logit_scale.half()
            else:
                logit_scale = self.myModel.model.logit_scalef

            logits_v2t = torch.cat([l_pos, l_neg_v2t], dim=-1) * logit_scale

            

            # anomaly_maps_myModel = []
            # anomaly_map = torch.softmax(feature_map1, dim=1)
            # anomaly_maps.append(anomaly_map)
            B, L, C = logits_v2t.shape
            H = int(np.sqrt(L))
            anomaly_map_myModel = F.interpolate(logits_v2t.permute(0, 2, 1).view(B, -1, H, H),
                                                size=224, mode='bilinear', align_corners=True)
            anomaly_map_myModel = torch.softmax(anomaly_map_myModel, dim=1)
            
            anomaly_map_all_myModel = anomaly_map_myModel[:, 1, :, :].unsqueeze(1)
            anomaly_map_prompts_myModel = self.prompt_learner1(anomaly_map_all_myModel)
            del anomaly_map_all_myModel,anomaly_map_myModel,logits_v2t,img_embeds, _, patch_tokens,noNeed2, feature_map1, feature_map2
            
            input_ids, target_ids, attention_mask = process_batch_instance(self.llama_tokenizer, output_texts,
                                                                           self.max_tgt_len)
            img_embeds_myModel, targets_myModel, attention_mask_myModel = self.prompt_wrap(img_embeds_myModel,
                                                                                           input_ids, target_ids,
                                                                                           attention_mask,
            anomaly_map_prompts_myModel)

            outputs = self.llama_model(
                inputs_embeds=img_embeds_myModel,
                attention_mask=attention_mask_myModel,
                return_dict=True,
                labels=targets_myModel,
            )
            loss = outputs.loss
            # calculate the token accuarcy
            chosen_tokens = torch.max(outputs.logits, dim=-1)[1][:, 1:-1]  # [B, S-1]
            labels = targets_myModel[:, 2:]
            gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(torch.long)  # [B*S]
            valid_mask = (labels != -100).reshape(-1)
            valid_tokens = gen_acc & valid_mask  # [B*S]
            gen_acc = valid_tokens.sum().item() / valid_mask.sum().item()

            return loss, gen_acc
            
            
            # =============================
            


    def extract_multimodal_feature(self, inputs, web_demo):
        features = []
        if inputs['image_paths']:
            
            prompt = inputs['prompt']
            c_name = 'object'
            for name in CLASS_NAMES:
                if name in prompt:
                    c_name = name
                    break
                
            if not web_demo:
                image_embeds, _, patch_tokens,image_embeds_myModel,noNeed2, feature_map1, feature_map2 = self.encode_image(inputs['image_paths'])
                feats_text_tensor = encode_text_with_prompt_ensemble(self.visual_encoder, [c_name], self.device)
            else:
                image_embeds, _, patch_tokens,image_embeds_myModel,noNeed2, feature_map1, feature_map2  = self.encode_image_for_web_demo(inputs['image_paths'])
                feats_text_tensor = encode_text_with_prompt_ensemble(self.visual_encoder, [c_name], self.device)
                
                
            normal_text_prompt, abnormal_text_prompt_handle, abnormal_text_prompt_learned = self.myModel.prompt_learner(
                c_name)
            normal_text_features = self.myModel.encode_text_embedding(normal_text_prompt,
                                                                      self.myModel.prompt_learner.tokenized_normal_prompts)
            abnormal_text_features_handle = self.myModel.encode_text_embedding(abnormal_text_prompt_handle,
                                                                               self.myModel.prompt_learner.tokenized_abnormal_prompts_handle)
            abnormal_text_features_learned = self.myModel.encode_text_embedding(abnormal_text_prompt_learned,
                                                                                self.myModel.prompt_learner.tokenized_abnormal_prompts_learned)
            abnormal_text_features = torch.cat([abnormal_text_features_handle, abnormal_text_features_learned], dim=0)
            # compute mean
            mean_ad_handle = torch.mean(F.normalize(abnormal_text_features_handle, dim=-1), dim=0)
            mean_ad_learned = torch.mean(F.normalize(abnormal_text_features_learned, dim=-1), dim=0)
            loss_match_abnormal = (mean_ad_handle - mean_ad_learned).norm(dim=0) ** 2.0
            # compute v2t loss and triplet loss
            normal_text_features_ahchor = normal_text_features.mean(dim=0).unsqueeze(0)
            normal_text_features_ahchor = normal_text_features_ahchor / normal_text_features_ahchor.norm(dim=-1,
                                                                                                         keepdim=True)
            abnormal_text_features_ahchor = abnormal_text_features.mean(dim=0).unsqueeze(0)
            abnormal_text_features_ahchor = abnormal_text_features_ahchor / abnormal_text_features_ahchor.norm(dim=-1,
                                                                                                               keepdim=True)
            abnormal_text_features = abnormal_text_features / abnormal_text_features.norm(dim=-1, keepdim=True)
            l_pos = torch.einsum('nic,cj->nij', noNeed2, normal_text_features_ahchor.transpose(0, 1))
            l_neg_v2t = torch.einsum('nic,cj->nij', noNeed2, abnormal_text_features.transpose(0, 1))
            if self.myModel.precision == 'fp16':
                logit_scale = self.myModel.model.logit_scale.half()
            else:
                logit_scale = self.myModel.model.logit_scalef

            logits_v2t = torch.cat([l_pos, l_neg_v2t], dim=-1) * logit_scale

            B, L, C = logits_v2t.shape
            H = int(np.sqrt(L))
            anomaly_map_myModel = F.interpolate(logits_v2t.permute(0, 2, 1).view(B, -1, H, H),
                                                size=224, mode='bilinear', align_corners=True)
            anomaly_map_myModel = torch.softmax(anomaly_map_myModel, dim=1)
            anomaly_map_all_myModel = anomaly_map_myModel[:, 1, :, :].unsqueeze(1)    
                
            

            anomaly_maps = []
            for layer in range(len(patch_tokens)):
                patch_tokens[layer] = patch_tokens[layer] / patch_tokens[layer].norm(dim=-1, keepdim=True)
                anomaly_map = (100.0 * patch_tokens[layer] @ feats_text_tensor.transpose(-2,-1))
                B, L, C = anomaly_map.shape
                H = int(np.sqrt(L))
                anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                            size=224, mode='bilinear', align_corners=True)
                anomaly_map = torch.softmax(anomaly_map, dim=1)
                anomaly_maps.append(anomaly_map[:,1,:,:])

            anomaly_map_ret = torch.mean(torch.stack(anomaly_maps, dim=0), dim=0).unsqueeze(1)
            if inputs['normal_img_paths']:
                query_patch_tokens , query_patch_map1, query_patch_map2= self.encode_image_for_one_shot(inputs['image_paths'])
                if 'mvtec' in 'normal_img_paths':
                    normal_patch_tokens , normal_patch_map1, normal_patch_map2= self.encode_image_for_one_shot_with_aug(inputs['normal_img_paths'])
                else:
                    normal_patch_tokens , normal_patch_map1, normal_patch_map2 = self.encode_image_for_one_shot(inputs['normal_img_paths'])
                sims = []
                # for i in range(len(query_patch_tokens)):
                #     query_patch_tokens_reshaped = query_patch_tokens[i].view(256, 1, 1024)
                #     normal_tokens_reshaped = normal_patch_tokens[i].reshape(1, -1, 1024)
                #     cosine_similarity_matrix = F.cosine_similarity(query_patch_tokens_reshaped, normal_tokens_reshaped,
                #                                                    dim=2)
                #     sim_max, _ = torch.max(cosine_similarity_matrix, dim=1)
                #     sims.append(sim_max)
                #
                # sim = torch.mean(torch.stack(sims, dim=0), dim=0).reshape(1, 1, 16, 16)
                # sim = F.interpolate(sim, size=224, mode='bilinear', align_corners=True)
                # anomaly_map_ret = 1 - sim  # (anomaly_map_ret + 1 - sim) / 2

                B = 1
                query_patch_map1 = query_patch_map1.view(B, 225, 1, 896)
                query_patch_map2 = query_patch_map2.view(B, 225, 1, 896)
                normal_patch_map1 = normal_patch_map1.view(B, 1, -1, 896)
                normal_patch_map2 = normal_patch_map2.view(B, 1, -1, 896)
                # 2,225,900
                cosine_similarity_matrix1 = F.cosine_similarity(query_patch_map1, normal_patch_map1,
                                                                dim=-1)
                cosine_similarity_matrix2 = F.cosine_similarity(query_patch_map2, normal_patch_map2,
                                                                dim=-1)
                # 2，225
                sim_max1, _ = torch.max(cosine_similarity_matrix1, dim=-1)
                sim_max2, _ = torch.max(cosine_similarity_matrix2, dim=-1)
                sim_myModel = torch.mean(torch.stack([sim_max1, sim_max2], dim=0), dim=0).reshape(B, 1, 15, 15)
                sim_myModel = F.interpolate(sim_myModel, size=224, mode='bilinear', align_corners=True)
            anomaly_map_all_myModel = 1 - sim_myModel
                

            features.append(image_embeds)
        if inputs['audio_paths']:
            audio_embeds, _ = self.encode_audio(inputs['audio_paths'])
            features.append(audio_embeds)
        if inputs['video_paths']:
            video_embeds, _ = self.encode_video(inputs['video_paths'])
            features.append(video_embeds)
        if inputs['thermal_paths']:
            thermal_embeds, _ = self.encode_thermal(inputs['thermal_paths'])
            features.append(thermal_embeds)

        feature_embeds = torch.cat(features).sum(dim=0).unsqueeze(0)
        return feature_embeds, anomaly_map_ret,anomaly_map_all_myModel,image_embeds_myModel

    def prepare_generation_embedding(self, inputs, web_demo):
        prompt = inputs['prompt']
        # if len(inputs['modality_embeds']) == 1:
        #     feature_embeds = inputs['modality_embeds'][0]
        # else:
        feature_embeds, anomaly_map,anomaly_map_myModel,feature_embeds_myModel = self.extract_multimodal_feature(inputs, web_demo)
        # print(anomaly_map.shape)
        feature_embeds = feature_embeds_myModel
        anomaly_map = anomaly_map_myModel
        # print(anomaly_map.shape)
        inputs['modality_embeds'].append(feature_embeds)

        batch_size = feature_embeds.shape[0]
        p_before = PROMPT_START
        p_before_tokens = self.llama_tokenizer(p_before, 
            return_tensors="pt", add_special_tokens=False).to(self.device)
        p_before_embeds = self.llama_model.model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1) # bsz x s1 x embed_dim
        
        p_middle = '</Img> '
        p_middle_tokens = self.llama_tokenizer(p_middle, 
            return_tensors="pt", add_special_tokens=False).to(self.device)
        # peft model need deeper call
        p_middle_embeds = self.llama_model.model.model.embed_tokens(p_middle_tokens.input_ids).expand(batch_size, -1, -1) # bsz x s1 x embed_dim

        # self.prompt_learner.eval()
        anomaly_map_prompts = self.prompt_learner1(anomaly_map)




        text = prompt + '\n### Assistant:'
        p_after_tokens = self.llama_tokenizer(text, add_special_tokens=False, return_tensors='pt').to(self.device)
        p_after_embeds = self.llama_model.model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1) # bsz x s2 x embed_dim
        bos = torch.ones([batch_size, 1],
                         dtype=p_before_tokens.input_ids.dtype,
                         device=p_before_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id # bsz x 1
        bos_embeds = self.llama_model.model.model.embed_tokens(bos) # bsz x 1 x embed_dim
        inputs_embeds = torch.cat([bos_embeds, p_before_embeds, feature_embeds, p_middle_embeds, anomaly_map_prompts, p_after_embeds], dim=1) # bsz x (1+s1+1+s2) x embed_dim
    
        return inputs_embeds, anomaly_map

    def generate(self, inputs, web_demo=False):
        '''
            inputs = {
                'image_paths': optional,
                'audio_paths': optional
                'video_paths': optional
                'thermal_paths': optional
                'mode': generation mode,
                'prompt': human input prompt,
                'max_tgt_len': generation length,
                'top_p': top_p,
                'temperature': temperature
                'modality_embeds': None or torch.tensor
                'modality_cache': save the image cache
            }
        '''
        # self.prompt_learner.eval()
        # self.llama_model.eval()
        # self.llama_proj.eval()
        # self.image_decoder.eval()
        # self.llama_tokenizer.eval()
        input_embeds, pixel_output = self.prepare_generation_embedding(inputs, web_demo)
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=[2277], encounters=1)])
        outputs = self.llama_model.generate(
            inputs_embeds=input_embeds,
            max_new_tokens=inputs['max_tgt_len'],
            top_p=inputs['top_p'],
            temperature=inputs['temperature'],
            do_sample=True,
            use_cache=True,
            stopping_criteria=stopping_criteria,
        )
        output_text = self.llama_tokenizer.decode(outputs[0][:-2], skip_special_tokens=True)
        return output_text, pixel_output