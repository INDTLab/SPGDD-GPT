from header import *
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from collections import OrderedDict
class DeepSpeedAgent:
    
    def __init__(self, model, args):
        super(DeepSpeedAgent, self).__init__()
        self.args = args
        self.model = model
        self.load_stage_1_parameters(args["delta_ckpt_path"])


        myModelPara = []
        for name, param in self.model.named_parameters():
            if(name.startswith('myModel')):
                myModelPara.append(f"{name} requires_grad: {param.requires_grad}")
            else:
                param.requires_grad = False
        #print(myModelPara)
        for name, param in self.model.image_decoder.named_parameters():
            param.requires_grad = True

        for name, param in self.model.prompt_learner1.named_parameters():
            param.requires_grad = True
            

        self.model.weights_cos_nolAndAbnol.requires_grad = True




        # load config parameters of deepspeed
        ds_params = json.load(open(self.args['ds_config_path']))
        ds_params['scheduler']['params']['total_num_steps'] = self.args['total_steps']
        ds_params['scheduler']['params']['warmup_num_steps'] = max(10, int(self.args['total_steps'] * self.args['warmup_rate']))
        self.ds_engine, self.optimizer, _ , _ = deepspeed.initialize(
            model=self.model, 
            model_parameters=self.model.parameters(),
            config_params=ds_params, 
            dist_init_required=True,
            args=types.SimpleNamespace(**args)
        )

    @torch.no_grad()
    def predict(self, batch):
        self.model.eval()
        string = self.model.generate_one_sample(batch)
        return string

    def train_model(self, batch, current_step=0, pbar=None):
        self.ds_engine.module.train()
        torch.cuda.empty_cache()
        loss, mle_acc = self.ds_engine(batch)
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"in {batch['img_paths']}find loss nan")
            
        

        self.ds_engine.backward(loss)
        
        torch.nn.utils.clip_grad_norm_(self.ds_engine.parameters(), max_norm=2.0)
        self.ds_engine.step()
        
        try:

            self.model.myModel.build_text_feature_gallery(batch['class_names'][0])
        except KeyError:

            print("no build_text_feature_gallery")
        pbar.set_description(f'[!] loss: {round(loss.item(), 4)}; token_acc: {round(mle_acc*100, 2)}')
        pbar.update(1)
        if self.args['local_rank'] == 0 and self.args['log_path'] and current_step % self.args['logging_step'] == 0:
            elapsed = pbar.format_dict['elapsed']
            rate = pbar.format_dict['rate']
            remaining = (pbar.total - pbar.n) / rate if rate and pbar.total else 0
            remaining = str(datetime.timedelta(seconds=remaining))
            logging.info(f'[!] progress: {round(pbar.n/pbar.total, 5)}; remaining time: {remaining}; loss: {round(loss.item(), 4)}; token_acc: {round(mle_acc*100, 2)}')
            
        mle_acc *= 100
        return mle_acc

    def save_model(self, path, current_step):
        step = current_step 
        
        checkpoint_dir = f'{path}checkpoints-60epochTrue'

 
        self.ds_engine.save_checkpoint(checkpoint_dir, step)

  
        fp32_state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir)

        print("Parameters in fp32_state_dict:")
        print("==============")


        param_grad_dic = {k: v.requires_grad for k, v in self.model.named_parameters()}
        filtered_state_dict = {k: v for k, v in fp32_state_dict.items() if param_grad_dic.get(k, False)}
        print("Parameters in filtered_state_dict:")
        print("==============")
   
        for k in param_grad_dic.keys():
            if k not in fp32_state_dict:
                print(f"Key not found in state_dict: {k}")


        if not os.path.exists(f'{path}{current_step}-60epochTrue'):
 
            os.makedirs(f'{path}{current_step}-60epochTrue')
            
            print(f" {path} create")
        
        torch.save(filtered_state_dict, f'{path}{current_step}-60epochTrue/pytorch_model_filtered.pt')
      
        # save tokenizer
        self.model.llama_tokenizer.save_pretrained(path)
        # save configuration
        self.model.llama_model.config.save_pretrained(path)
        print(f'[!] save model into {path}')

    def load_stage_1_parameters(self, path):
        delta_ckpt = torch.load(path, map_location=torch.device('cpu'))
        self.model.load_state_dict(delta_ckpt, strict=False)
