

import torch
import sys
from transformer.transformer import Transformer
import yaml
from data_utils import  genSpoof_list, Dataset_ASVspoof
from torch.utils.data import DataLoader
import main_transformer
from main_rawnet import produce_evaluation_transformer_file
lr=0.0001

eval_output = r'I:\2021\outputs\output_transformer.txt'
dir_yaml = r'I:\2021\PA\Baseline-RawNet2\model_config_Transformer.yaml'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = r'I:\2021\PA\Baseline-RawNet2\models\model_Transformer_PA_weighted_CCE_1_6_0.0001\epoch_0.pth'
with open(dir_yaml, 'r') as f_yaml:
    parser1 = yaml.load(f_yaml, yaml.FullLoader)
model = Transformer(parser1['model'])
nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
model = (model).to(device)

# set Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=lr)
model.load_state_dict(torch.load(model_path, map_location=device))
print('Model loaded : {}'.format(model_path))

files_meta_eval = genSpoof_list(is_logical=False, is_train=False,
                                is_eval=True)
eval_set = Dataset_ASVspoof(files_meta_eval)
data_loader = DataLoader(eval_set, batch_size=6, shuffle=False, drop_last=False)
del files_meta_eval,eval_set
produce_evaluation_transformer_file(data_loader, model, device, eval_output)
sys.exit(0)
