import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import warnings
warnings.filterwarnings('ignore')
import argparse
import logging
import torch.nn as nn
from torch.nn.modules.loss import CrossEntropyLoss
import torch.optim as optim
from torchvision import transforms
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from utils import ramps, losses, metrics, test_patch
from dataloaders.dataset import *
from networks.net_factory import net_factory
from utils.mixmatch_util import mix_module


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='Pancreas_CT', help='Pancreas_CT,LA')
parser.add_argument('--root_path', type=str, default='../', help='Name of Dataset')
parser.add_argument('--exp', type=str, default='debug', help='exp_name')
parser.add_argument('--model', type=str, default='VNet_4out', help='model_name')
parser.add_argument('--max_iteration', type=int, default=15000, help='maximum iteration to train')
parser.add_argument('--max_samples', type=int, default=80, help='maximum samples to train')
parser.add_argument('--labeled_bs', type=int, default=2, help='batch_size of labeled data per gpu')  # 2
parser.add_argument('--batch_size', type=int, default=4, help='batch_size of labeled data per gpu')  # 4
parser.add_argument('--base_lr', type=float, default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int, default=6, help='trained samples')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--consistency', type=float, default=1, help='consistency_weight')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
parser.add_argument('--temperature', type=float, default=0.1, help='temperature of sharpening')
parser.add_argument('--lamda', type=float, default=0.5, help='weight to balance all losses')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
snapshot_path = args.root_path + "model/{}_{}_{}_labeled/{}".format(args.dataset_name, args.exp, args.labelnum,
                                                                    args.model)

num_classes = 2
if args.dataset_name == "LA":
    # patch_size = (32, 32, 32)   # for debug use, quickly the training process
    patch_size = (112, 112, 80)
    args.root_path = args.root_path + 'data/LA'
    args.max_samples = 80
    args.max_iteration = 15000
elif args.dataset_name == "Pancreas_CT":
    patch_size = (96, 96, 96)
    # patch_size = (32, 32, 32)
    args.root_path = args.root_path + 'data/Pancreas'
    args.max_samples = 62
    args.max_iteration = 15000

train_data_path = args.root_path

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
labeled_bs = args.labeled_bs
max_iterations = args.max_iteration
base_lr = args.base_lr

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)
def sharpening(P):
    T = 1 / args.temperature
    P_sharpen = P ** T / (P ** T + (1 - P) ** T)
    return P_sharpen

class WeightEMA(object):
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * args.base_lr

        # Initialize the parameters of both models to the same value first
        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        # Afterwards, perform smooth updates on the ema model every time
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype == torch.float32:
                # 0.99 * previous model parameters+0.01 * updated model parameters
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)
                # customized weight decay
                param.mul_(1 - self.wd)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    # shutil.copytree('../code/', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))


    def create_model(ema=False):
        # Network definition
        net = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    # init the model, include the teacher network and student network
    model = create_model(ema=False)         #student network
    ema_model = create_model(ema=True)      #teacher network

    # init the dataset
    if args.dataset_name == "LA":
        db_train = LAHeart_no_read(base_dir=train_data_path,
                           split='train',
                           transform=transforms.Compose([
                               RandomRotFlip(),
                               RandomCrop(patch_size),
                               ToTensor(),
                           ]))
    elif args.dataset_name == "Pancreas_CT":
        db_train = Pancreas_no_read(base_dir=train_data_path,
                            split='train',
                            transform=transforms.Compose([
                                RandomCrop(patch_size),
                                ToTensor(),
                            ]))
    #set the labeled num selection for the training dataset
    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size - labeled_bs)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)


    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    ema_optimizer = WeightEMA(model, ema_model, alpha=0.99)
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))
    MSE_cri = losses.mse_loss

    iter_num,best_dice = 0,0
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    kl_distance = nn.KLDivLoss(reduction='none')
    max_epoch = max_iterations // len(trainloader) + 1
    lr_ = base_lr
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            l_image,l_label = volume_batch[:args.labeled_bs],label_batch[:args.labeled_bs]
            unlabeled_volume_batch = volume_batch[args.labeled_bs:]

            X = list(zip(l_image, l_label))
            U = unlabeled_volume_batch

            X_prime, U_prime, pseudo_label = mix_module(X, U, eval_net=ema_model, K=2, T=0.5, alpha=0.75,
                                                         mixup_mode='_x', aug_factor=torch.tensor(1).cuda())
            model.train()
            X_data = torch.cat([torch.unsqueeze(X_prime[0][0],0), torch.unsqueeze(X_prime[1][0],0)],0) #  需要unsqueeze 一下
            X_label = torch.cat([torch.unsqueeze(X_prime[0][1], 0), torch.unsqueeze(X_prime[1][1], 0)],0)
            U_data = torch.cat([torch.unsqueeze(U_prime[0][0], 0), torch.unsqueeze(U_prime[1][0], 0),
                                torch.unsqueeze(U_prime[2][0], 0),torch.unsqueeze(U_prime[3][0], 0)],0)

            X_data ,X_label = X_data.cuda(),X_label.cuda().float()
            U_data ,U_data_pseudo = U_data.cuda(),pseudo_label.cuda().float()

            X = torch.cat((X_data, U_data,volume_batch[args.labeled_bs:]), 0)

            out_1_all, out_2_all,out_3_all,out_4_all = model(X)

            out_1,out_2,out_3,out_4             = \
                out_1_all[:-args.labeled_bs],out_2_all[:-args.labeled_bs],out_3_all[:-args.labeled_bs],out_4_all[:-args.labeled_bs]
            out_1_u,out_2_u,out_3_u,out_4_u     = \
                out_1_all[-args.labeled_bs:],out_2_all[-args.labeled_bs:],out_3_all[-args.labeled_bs:],out_4_all[-args.labeled_bs:]
            out_1_s,out_2_s,out_3_s,out_4_s         = \
                torch.softmax(out_1, dim=1), torch.softmax(out_2, dim=1),torch.softmax(out_3, dim=1),torch.softmax(out_4, dim=1)
            out_1_u_s,out_2_u_s,out_3_u_s,out_4_u_s = \
                torch.softmax(out_1_u, dim=1), torch.softmax(out_2_u, dim=1), torch.softmax(out_3_u, dim=1),torch.softmax(out_4_u, dim=1)

            o_1_u_s = torch.softmax(out_1_all[args.labeled_bs:], dim=1)
            o_2_u_s = torch.softmax(out_2_all[args.labeled_bs:], dim=1)
            o_3_u_s = torch.softmax(out_3_all[args.labeled_bs:], dim=1)
            o_4_u_s = torch.softmax(out_4_all[args.labeled_bs:], dim=1)
            loss_seg_ce_lab, loss_seg_ce_unlab = 0, 0
            loss_seg_dice_lab, loss_seg_dice_unlab = 0, 0

            loss_seg_ce_lab += ce_loss(out_1[:args.labeled_bs], X_label[:args.labeled_bs].long()) + \
                               ce_loss(out_2[:args.labeled_bs], X_label[:args.labeled_bs].long()) + \
                               ce_loss(out_3[:args.labeled_bs], X_label[:args.labeled_bs].long())+\
                               ce_loss(out_4[:args.labeled_bs], X_label[:args.labeled_bs].long())

            loss_seg_dice_lab += dice_loss(out_1_s[:args.labeled_bs],X_label[:args.labeled_bs].unsqueeze(1)) + \
                                 dice_loss(out_2_s[:args.labeled_bs], X_label[:args.labeled_bs].unsqueeze(1)) + \
                                 dice_loss(out_3_s[:args.labeled_bs], X_label[:args.labeled_bs].unsqueeze(1)) + \
                                 dice_loss(out_4_s[:args.labeled_bs], X_label[:args.labeled_bs].unsqueeze(1))

            loss_seg_ce_unlab += ce_loss(out_1[args.labeled_bs:], U_data_pseudo[:].long()) + \
                                 ce_loss(out_2[args.labeled_bs:], U_data_pseudo[:].long()) +\
                                 ce_loss(out_3[args.labeled_bs:], U_data_pseudo[:].long()) + \
                                 ce_loss(out_4[args.labeled_bs:], U_data_pseudo[:].long())

            loss_seg_dice_unlab += dice_loss(out_1_s[args.labeled_bs:], U_data_pseudo[:].unsqueeze(1)) + \
                                   dice_loss(out_2_s[args.labeled_bs:], U_data_pseudo[:].unsqueeze(1)) + \
                                   dice_loss(out_3_s[args.labeled_bs:], U_data_pseudo[:].unsqueeze(1)) + \
                                   dice_loss(out_4_s[args.labeled_bs:], U_data_pseudo[:].unsqueeze(1))

            supervised_loss = 0.5 * (loss_seg_ce_lab + loss_seg_dice_lab)
            pseudo_loss = 0.5 * (loss_seg_dice_unlab + loss_seg_ce_unlab)
            preds = (o_1_u_s + o_2_u_s + o_3_u_s + o_4_u_s) / 4

            variance_1 = torch.sum(kl_distance(torch.log(o_1_u_s), preds), dim=1, keepdim=True)# 只是用来计算kl，固定操作，多加一个log
            exp_variance_1 = torch.exp(-variance_1)
            variance_2 = torch.sum(kl_distance(torch.log(o_2_u_s), preds), dim=1, keepdim=True)
            exp_variance_2 = torch.exp(-variance_2)
            variance_3 = torch.sum(kl_distance(torch.log(o_3_u_s), preds), dim=1, keepdim=True)
            exp_variance_3 = torch.exp(-variance_3)
            variance_4 = torch.sum(kl_distance(torch.log(o_4_u_s), preds), dim=1, keepdim=True)
            exp_variance_4 = torch.exp(-variance_4)

            consis_dist_1 = (preds - o_1_u_s) ** 3
            consis_loss_1 = torch.mean(consis_dist_1 * exp_variance_1) / (torch.mean(exp_variance_1) + 1e-8) + torch.mean(variance_1)
            consis_dist_2 = (preds - o_2_u_s) ** 3
            consis_loss_2 = torch.mean(consis_dist_2 * exp_variance_2) / (torch.mean(exp_variance_2) + 1e-8) + torch.mean( variance_2)
            consis_dist_3 = ( preds - o_3_u_s) ** 3
            consis_loss_3 = torch.mean(consis_dist_3 * exp_variance_3) / (torch.mean(exp_variance_3) + 1e-8) + torch.mean(variance_3)
            consis_dist_4 = (preds - o_4_u_s) ** 3
            consis_loss_4 = torch.mean(consis_dist_4 * exp_variance_4) / (
                        torch.mean(exp_variance_4) + 1e-8) + torch.mean(variance_4)

            sharp1 = sharpening(out_1_u_s)
            sharp2 = sharpening(out_2_u_s)
            sharp3 = sharpening(out_3_u_s)
            sharp4 = sharpening(out_4_u_s)

            loss_consist =  (consis_loss_1 +  consis_loss_2 + consis_loss_3 + consis_loss_4)/4 \
                            +(MSE_cri(sharp1,out_1_u_s) + MSE_cri(sharp2,out_2_u_s) +
                               MSE_cri(sharp3,out_3_u_s)+MSE_cri(sharp4,out_4_u_s))/4

            consistency_weight = get_current_consistency_weight(iter_num // 150)
            loss = supervised_loss + pseudo_loss + consistency_weight * loss_consist

            iter_num = iter_num + 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ema_optimizer.step()
            update_ema_variables(model, ema_model, 0.99, iter_num)
            consistency_loss1 = 0
            if iter_num % 100 == 0:
                logging.info('iteration %d : loss : %03f, loss_d: %03f, loss_cosist: %03f' % (
                    iter_num, loss, supervised_loss, loss_consist))

            writer.add_scalar('Labeled_loss/loss_seg_dice', loss_seg_dice_lab, iter_num)
            writer.add_scalar('Labeled_loss/pseudo_loss', pseudo_loss, iter_num)
            writer.add_scalar('Labeled_loss/loss_seg_ce', loss_seg_ce_lab, iter_num)
            writer.add_scalar('Co_loss/consistency_loss', loss_consist, iter_num)
            writer.add_scalar('Co_loss/consist_weight', consistency_weight, iter_num)
            if iter_num % 500 == 0:
                image = volume_batch[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                image = torch.argmax(out_1_s, dim=1, keepdim=True)[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1) * 100
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_label',
                                 grid_image, iter_num)

                image = label_batch[0, :, :, 20:61:10].unsqueeze(
                    0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1) * 100
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Groundtruth_label',
                                 grid_image, iter_num)

            if  iter_num % 200 == 0:
                model.eval()
                if args.dataset_name == "LA":
                    dice_sample = test_patch.var_all_case(model, num_classes=num_classes, patch_size=patch_size,
                                                          stride_xy=18, stride_z=4, dataset_name='LA')
                elif args.dataset_name == "Pancreas_CT":
                    dice_sample = test_patch.var_all_case(model, num_classes=num_classes, patch_size=patch_size,
                                                          stride_xy=32, stride_z=32, dataset_name='Pancreas_CT')
                if dice_sample > best_dice:
                    best_dice = dice_sample
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)
                    logging.info("save best model to {}".format(save_mode_path))
                writer.add_scalar('Var_dice/Dice', dice_sample, iter_num)
                writer.add_scalar('Var_dice/Best_dice', best_dice, iter_num)
                model.train()

            if iter_num >= max_iterations:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
                print('best_dice',best_dice)
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
