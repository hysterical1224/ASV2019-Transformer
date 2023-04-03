import argparse
import sys
import os
import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
import yaml
# from data_utils import genSpoof_list,Dataset_ASVspoof2019_train,Dataset_ASVspoof2021_eval, Dataset_ASVspoof2019
from data_utils import genSpoof_list, Dataset_ASVspoof
from losses import FocalLoss
from model.raw_res2net import Res2Net
from tensorboardX import SummaryWriter
from core_scripts.startup_config import set_random_seed

__author__ = "Shen Song"
__email__ = "1403826619@qq.com"

sysid_dict = {
    '-': 0,  # bonafide speech
    'SS_1': 1,  # Wavenet vocoder
    'SS_2': 2,  # Conventional vocoder WORLD
    'SS_4': 3,  # Conventional vocoder MERLIN
    'US_1': 4,  # Unit selection system MaryTTS
    'VC_1': 5,  # Voice conversion using neural networks
    'VC_4': 6,  # transform function-based voice conversion
    # For PA:
    'AA': 7,
    'AB': 8,
    'AC': 9,
    'BA': 10,
    'BB': 11,
    'BC': 12,
    'CA': 13,
    'CB': 14,
    'CC': 15
}


def evaluate_accuracy(dev_loader, model, device):
    num_correct = 0.0
    num_total = 0.0

    for batch_x, batch_y, batch_meta in dev_loader:
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        model.eval()
        batch_out = model(batch_x)
        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
    return 100 * (num_correct / num_total)


def produce_evaluation_transformer_file(data_loader, model, device, save_path):
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    true_y = []
    fname_list = []
    key_list = []
    sys_id_list = []
    key_list = []
    score_list = []
    sysid_dict_inv = dict(zip(sysid_dict.values(), sysid_dict.keys()))
    for batch_x, batch_y, batch_meta in data_loader:
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_out = model(batch_x)
        batch_score = (batch_out[:, 1] - batch_out[:, 0]
                       ).data.cpu().numpy().ravel()

        # add outputs
        fname_list.extend(list(batch_meta[1]))
        key_list.extend(
            ['bonafide' if key == 1 else 'spoof' for key in list(batch_meta[4])])
        sys_id_list.extend([sysid_dict_inv[s.item()]
                            for s in list(batch_meta[3])])
        score_list.extend(batch_score.tolist())

    with open(save_path, 'w') as fh:
        for f, s, k, cm in zip(fname_list, sys_id_list, key_list, score_list):
            fh.write('{} {} {} {}\n'.format(f, s, k, cm))
    print('Result saved to {}'.format(save_path))


def produce_evaluation_file(dataset, model, device, save_path):
    data_loader = DataLoader(dataset, batch_size=6, shuffle=False, drop_last=False)
    # datameta:seaker id,path,sysid,key
    num_total = 0.0
    num_correct = 0.0
    model.eval()
    fname_list = []
    sys_id_list = []
    key_list = []
    score_list = []
    for batch_x, batch_y, batch_meta in data_loader:
        # print('batch_meta',batch_meta)
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_out = model(batch_x)
        batch_score = (batch_out[:, 1] - batch_out[:, 0]
                       ).data.cpu().numpy().ravel()
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
        print(100 * (num_correct / num_total))
        # add outputs

        fname_list.extend(list(batch_meta[1]))
        key_list.extend(
            ['bonafide' if key == 1 else 'spoof' for key in list(batch_meta[4])])
        sys_id_list.extend([dataset.sysid_dict_inv[s.item()]
                            for s in list(batch_meta[3])])
        score_list.extend(batch_score.tolist())

    with open(save_path, 'w') as fh:
        for f, s, k, cm in zip(fname_list, sys_id_list, key_list, score_list):
            fh.write('{} {} {} {}\n'.format(f, s, k, cm))
    print('Result saved to {}'.format(save_path))


def train_epoch(train_loader, model, optim, device):
    running_loss = 0
    num_correct = 0.0
    num_total = 0.0
    ii = 0

    # set objective (Loss) functions
    # weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = FocalLoss()

    for batch_x, batch_y, batch_meta in train_loader:
        batch_size = batch_x.size(0)
        num_total += batch_size
        ii += 1
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        model.train()
        batch_out = model(batch_x)
        batch_loss = criterion(batch_out, batch_y)
        _, batch_pred = batch_out.max(dim=1)

        num_correct += (batch_pred == batch_y).sum(dim=0).item()
        running_loss += (batch_loss.item() * batch_size)
        if ii % 10 == 0:
            sys.stdout.write('\r \t train:{:.2f}'.format(
                (num_correct / num_total) * 100))
        optim.zero_grad()
        batch_loss.backward()
        optim.step()

    running_loss /= num_total
    train_accuracy = (num_correct / num_total) * 100
    return running_loss, train_accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ASVspoof2021 baseline system')
    # Dataset
    parser.add_argument('--database_path', type=str, default=r'I:\2021\data\PA/',
                        help='Change this to user\'s full directory address of PA database (ASVspoof2019- for training & validation, ASVspoof2021 for evaluation scores). We assume that all three ASVspoof 2019 PA train, PA dev and ASVspoof2021 PA eval data folders are in the same database_path directory.')
    '''
    % database_path/
    %   |- PA
    %      |- ASVspoof2021_PA_eval/flac
    %      |- ASVspoof2019_PA_train/flac
    %      |- ASVspoof2019_PA_dev/flac
    '''

    parser.add_argument('--protocols_path', type=str, default=r'I:\2021\data\PA/',
                        help='Change with path to user\'s PA database protocols directory address')
    '''
    % protocols_path/
    %   |- ASVspoof_PA_cm_protocols
    %      |- ASVspoof2021.PA.cm.eval.trl.txt
    %      |- ASVspoof2019.PA.cm.dev.trl.txt 
    %      |- ASVspoof2019.PA.cm.train.trn.txt 
    '''

    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='weighted_CCE')
    # model
    parser.add_argument('--seed', type=int, default=134,
                        help='random seed (default: 1234)')

    parser.add_argument('--model_path', type=str,
                        default=r"I:\2021\PA\Baseline-RawNet2\run\models\model_PA_weighted_CCE_60_6_0.0001\iter_40.pth", help='Model checkpoint')
    parser.add_argument('--comment', type=str, default=None,
                        help='Comment to describe the saved model')
    # Auxiliary arguments
    parser.add_argument('--track', type=str, default='PA', choices=['LA', 'PA', 'DF'], help='LA/PA/DF')
    parser.add_argument('--eval_output', type=str, default=r'I:\2021\outputs\output_res2net.txt',
                        help='Path to save the evaluation result')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='eval mode')
    parser.add_argument('--is_eval', action='store_true', default=False, help='eval database')
    parser.add_argument('--eval_part', type=int, default=0)
    # backend options
    parser.add_argument('--cudnn-deterministic-toggle', action='store_false', \
                        default=True,
                        help='use cudnn-deterministic? (default true)')

    parser.add_argument('--cudnn-benchmark-toggle', action='store_true', \
                        default=False,
                        help='use cudnn-benchmark? (default false)')
    parser.add_argument('--save_path', type=str, default=r'I:\2021\outputs\output_res2net.txt')

    dir_yaml =r"I:\2021\PA\Baseline-RawNet2\config\model_config_res2net.yaml"

    with open(dir_yaml, 'r') as f_yaml:
        parser1 = yaml.load(f_yaml, yaml.FullLoader)

    if not os.path.exists('models'):
        os.mkdir('models')
    args = parser.parse_args()

    # make experiment reproducible
    set_random_seed(args.seed, args)

    track = args.track
    is_logical = (track == 'LA')

    assert track in ['LA', 'PA', 'DF'], 'Invalid track given'

    # Database

    prefix = 'ASVspoof2019_{}'.format(track)
    prefix_2019 = 'ASVspoof2019.{}'.format(track)
    prefix_2021 = 'ASVspoof2019.{}'.format(track)

    # define model saving path
    model_tag = 'model_{}_{}_{}_{}_{}'.format(
        track, args.loss, args.num_epochs, args.batch_size, args.lr)
    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)

    model_save_path = os.path.join("models", model_tag)

    # set model save directory
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    # GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: {}'.format(device))

    # model
    model = Res2Net(parser1['model'], device)
    # nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    # pretrained_params = torch.load(r"I:\2021\PA\Baseline-RawNet2\run\models\model_PA_weighted_CCE_60_6_0.0001\iter_40.pth", map_location=device)
    # model.load_state_dict(pretrained_params, strict=False)
    model = (model).to(device)
    # print("model:",list(model.parameters()))

    # set Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=device)['model_state_dict'])
        print('Model loaded : {}'.format(args.model_path))

    # define train dataloader

    # d_label_trn,file_train = genSpoof_list( dir_meta =  os.path.join(args.protocols_path+'{}_cm_protocols/{}.cm.train.trn.txt'.format(prefix,prefix_2019)),is_train=True,is_eval=False)
    # print('no. of training trials',len(file_train))

    # train_set=Dataset_ASVspoof2019_train(list_IDs = file_train,labels = d_label_trn,base_dir = os.path.join(args.database_path+'ASVspoof2019_{}_train/'.format(args.track)))
    files_meta_train = genSpoof_list(is_logical=is_logical, is_train=True,
                                     is_eval=args.is_eval)
    train_set = Dataset_ASVspoof(files_meta_train)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    del train_set, files_meta_train
    files_meta_dev = genSpoof_list(is_logical=is_logical, is_train=False, is_eval=args.is_eval)
    dev_set = Dataset_ASVspoof(files_meta_dev)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=False)
    # del dev_set,d_label_dev
    del dev_set, files_meta_dev

    # Training and validation
    num_epochs = args.num_epochs
    save_path = args.save_path
    writer = SummaryWriter('logs/{}'.format(model_tag))
    best_acc = 92
    # def produce_evaluation_file(dataset, model, device, save_path):
    for epoch in range(num_epochs):
        # print("model:", list(model.parameters()))
        running_loss, train_accuracy = train_epoch(train_loader, model, optimizer, device)
        valid_accuracy = evaluate_accuracy(dev_loader, model, device)
        # valid_accuracy = produce_evaluation_file(dev_set, model, device, save_path)
        writer.add_scalar('train_accuracy', train_accuracy, epoch)
        writer.add_scalar('valid_accuracy', valid_accuracy, epoch)
        writer.add_scalar('loss', running_loss, epoch)
        print('\n{} - {} - {:.2f} - {:.2f}'.format(epoch,
                                                   running_loss, train_accuracy, valid_accuracy))

        if valid_accuracy > best_acc:
            print('best model find at epoch', epoch)
            best_acc = max(valid_accuracy, best_acc)
            save_path = os.path.join(model_save_path,
                                     'iter_{}.pth'.format(epoch))
            checkpoint = {'model_state_dict': model.state_dict(),
                          'optimizer_state_dict': optimizer.state_dict(),
                          'rng_state': torch.get_rng_state(),
                          'cuda_rng_state': torch.cuda.get_rng_state()}
            torch.save(checkpoint, save_path)
    del dev_loader, train_loader
    # evaluation
