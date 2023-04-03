
from main_rawnet import produce_evaluation_file
import torch
import sys
from .model import RawNet
import yaml
from data_utils import  genSpoof_list, Dataset_ASVspoof
lr=0.0001
# 最好的模型：68
eval_output = r'I:\2021\outputs\output_NFFT_2048_epoch_68.txt'
dir_yaml = r'I:\2021\PA\Baseline-RawNet2\model_config_RawNet.yaml'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = r'I:\2021\PA\Baseline-RawNet2\models\model_PA_weighted_CCE_100_16_0.0001\epoch_68.pth'
with open(dir_yaml, 'r') as f_yaml:
    parser1 = yaml.load(f_yaml, yaml.FullLoader)
model =RawNet(parser1['model'], device)
nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
model = (model).to(device)

# set Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=lr)
model.load_state_dict(torch.load(model_path, map_location=device))
print('Model loaded : {}'.format(model_path))

files_meta_eval = genSpoof_list(is_logical=False, is_train=False,
                                is_eval=True)
eval_set = Dataset_ASVspoof(files_meta_eval)
del files_meta_eval
produce_evaluation_file(eval_set, model, device, eval_output)
sys.exit(0)
