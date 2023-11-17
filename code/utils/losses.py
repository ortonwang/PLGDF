import torch
import torch.nn as nn

def Binary_dice_loss(predictive, target, ep=1e-8):
    intersection = 2 * torch.sum(predictive * target) + ep
    union = torch.sum(predictive) + torch.sum(target) + ep
    loss = 1 - intersection / union
    return loss

def kl_loss(inputs, targets, ep=1e-8):
    kl_loss=nn.KLDivLoss(reduction='mean')
    consist_loss = kl_loss(torch.log(inputs+ep), targets)
    return consist_loss

class soft_ce_loss_o(nn.Module):
    def __init__(self, n_classes=2):
        super(soft_ce_loss_o, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(torch.unsqueeze(temp_prob,1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()
    def forward(self, inputs, target):
        # if softmax:
        inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        # logprobs = torch.log(inputs + 1e-8)
        final_outputs = inputs * target.detach()
        loss_vec = - ((final_outputs).sum(dim=1))
        # p = torch.exp(-loss_vec)
        # loss_vec =  (1 - p) ** self.gamma * loss_vec
        average_loss = loss_vec.mean()
        return average_loss


def soft_ce_loss(inputs, target, ep=1e-8):
    logprobs = torch.log(inputs+ep)
    return  torch.mean(-(target[:,0,...]*logprobs[:,0,...]+target[:,1,...]*logprobs[:,1,...]))

# aa = torch.zeros(4,2,96,96,96)
# bb = torch.zeros(4,96,96,96)
# cri = soft_ce_loss_o()
# loss = cri(aa,bb)
#
# print('fds')


def mse_loss(input1, input2):
    return torch.mean((input1 - input2)**2)


def reshape_tensor_to_2D(x):
    """
    Reshape input tensor of shape [N, C, D, H, W] or [N, C, H, W] to [voxel_n, C]
    """
    tensor_dim = len(x.size())
    num_class = list(x.size())[1]
    if (tensor_dim == 5):
        x_perm = x.permute(0, 2, 3, 4, 1)
    elif (tensor_dim == 4):
        x_perm = x.permute(0, 2, 3, 1)
    else:
        raise ValueError("{0:}D tensor not supported".format(tensor_dim))

    y = torch.reshape(x_perm, (-1, num_class))
    return y


def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-10
        intersection = torch.sum(score * target)
        union = torch.sum(score * score) + torch.sum(target * target) + smooth
        loss = 1 - intersection / union
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

class DiceLoss_o(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss_o, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-10
        intersection = torch.sum(score * target)
        union = torch.sum(score * score) + torch.sum(target * target) + smooth
        loss = 1 - intersection*5 / (union+ intersection*4)
        return loss

    def forward(self, inputs, target0, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target0)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


class NoiseRobustDiceLoss(nn.Module):

    def __init__(self):
        super(NoiseRobustDiceLoss, self).__init__()
        self.gamma = 1.5#params['NoiseRobustDiceLoss_gamma'.lower()]

    def forward(self, predict,soft_y):
        # predict = loss_input_dict['prediction']
        # soft_y = loss_input_dict['ground_truth']

        if (isinstance(predict, (list, tuple))):
            predict = predict[0]
        # if (self.softmax):
        #     predict = nn.Softmax(dim=1)(predict)
        predict = reshape_tensor_to_2D(predict)
        soft_y = reshape_tensor_to_2D(soft_y)

        numerator = torch.abs(predict - soft_y)
        numerator = torch.pow(numerator, self.gamma)
        denominator = predict + soft_y
        numer_sum = torch.sum(numerator, dim=0)
        denom_sum = torch.sum(denominator, dim=0)
        loss_vector = numer_sum / (denom_sum + 1e-5)
        loss = torch.mean(loss_vector)
        return loss
