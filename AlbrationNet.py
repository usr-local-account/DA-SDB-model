# -*- coding: utf-8 -*-

from typing import List, Dict
import torch.nn as nn
import torch

from tllib.modules.kernels import GaussianKernel
from tllib.alignment.dan import MultipleKernelMaximumMeanDiscrepancy
from tllib.alignment.jan import JointMultipleKernelMaximumMeanDiscrepancy
from tllib.alignment.coral import CorrelationAlignmentLoss
from tllib.alignment.dann import DomainAdversarialLoss
from tllib.modules.domain_discriminator import DomainDiscriminator
from tllib.modules.grl import WarmStartGradientReverseLayer

# In[0] add
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def DARE_GRAM_LOSS(H1, H2, treshold=0.9, tradeoff_angle=0.05, tradeoff_scale=0.001):    
    b,p = H1.shape

    A = torch.cat((torch.ones(b,1).to(device), H1), 1)
    B = torch.cat((torch.ones(b,1).to(device), H2), 1)

    cov_A = (A.t()@A)
    cov_B = (B.t()@B) 

    _,L_A,_ = torch.linalg.svd(cov_A)
    _,L_B,_ = torch.linalg.svd(cov_B)
    
    eigen_A = torch.cumsum(L_A.detach(), dim=0)/L_A.sum()
    eigen_B = torch.cumsum(L_B.detach(), dim=0)/L_B.sum()

    if(eigen_A[1]>treshold):
        T = eigen_A[1].detach()
    else:
        T = treshold
        
    index_A = torch.argwhere(eigen_A.detach()<=T)[-1]

    if(eigen_B[1]>treshold):
        T = eigen_B[1].detach()
    else:
        T = treshold

    index_B = torch.argwhere(eigen_B.detach()<=T)[-1]
    
    k = max(index_A, index_B)[0]

    A = torch.linalg.pinv(cov_A ,rtol = (L_A[k]/L_A[0]).detach())
    B = torch.linalg.pinv(cov_B ,rtol = (L_B[k]/L_B[0]).detach())
    
    cos_sim = nn.CosineSimilarity(dim=0,eps=1e-6)
    cos = torch.dist(torch.ones((p+1)).to(device),(cos_sim(A,B)),p=1)/(p+1)
    return tradeoff_angle*(cos) + tradeoff_scale*torch.dist((L_A[:k]),(L_B[:k]))/k

# In[0-] Create FullconnectBlock Model
class FullConnectBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_prob):
        super(FullConnectBlock, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout_prob = dropout_prob
        
        self.linear1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.dp1 = nn.Dropout(p=self.dropout_prob)
        self.linear2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.dp2 = nn.Dropout(p=self.dropout_prob)
        self.linear3 = nn.Linear(out_features, out_features)
        self.bn3 = nn.BatchNorm1d(out_features)
        self.dp3 = nn.Dropout(p=self.dropout_prob)
        
    def forward(self, x):
        output = self.linear1(x)
        output1 = torch.relu(self.bn1(output))
        output1 = self.dp1(output1)
        
        output2 = self.linear2(output1)
        output2 = torch.relu(self.bn2(output2))
        output2 = self.dp2(output2)
        
        output3 = self.linear3(output2)
        output3 = torch.relu(self.bn3(output3))
        output3 = self.dp3(output3)
        
        return output3
 # In[0+] Create FPyramidBlock Model
class FPyramidBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_prob):
        super(FPyramidBlock, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout_prob = dropout_prob
            
        self.channel1 = FullConnectBlock(in_features, out_features, dropout_prob)
        self.channel2 = FullConnectBlock(in_features, out_features*2, dropout_prob)
        self.channel3 = FullConnectBlock(in_features, out_features*4, dropout_prob)
        
        self.linear_connect1 = nn.Linear(out_features, out_features*2)
        self.bn1 = nn.BatchNorm1d(out_features*2)
        self.dp1 = nn.Dropout(p=self.dropout_prob)
        self.linear_connect2 = nn.Linear(out_features*2, out_features*4)
        self.bn2 = nn.BatchNorm1d(out_features*4)
        self.dp2 = nn.Dropout(p=self.dropout_prob)
    def forward(self, x):
        output1 = self.channel1(x)
        output2 = self.channel2(x)
        output3 = self.channel3(x)
        
        connected_output1 = self.linear_connect1(output1)
        connected_output1 = torch.relu(self.bn1(connected_output1))
        connected_output1 = self.dp1(connected_output1)
        
        output2 = output2 + connected_output1
        
        connected_output2 = self.linear_connect2(output2)
        connected_output2 = torch.relu(self.bn2(connected_output2))
        connected_output2 = self.dp2(connected_output2)
        
        output3 = output3 + connected_output2
        
        return output3   
        
# In[1] Create RegressorFPN Model
class RegressorFPN(nn.Module):
    def __init__(self, input_dims:int, out_features:int=16, dropout_prob:float=0.1):
        super(RegressorFPN, self).__init__()
        self.input_dims = input_dims 
        self.out_features = out_features 
        self.dropout_prob = dropout_prob
        
        self.hidden_FPyramidBlock1 = nn.Sequential(FPyramidBlock(self.input_dims, self.out_features, self.dropout_prob))
        self.hidden_FPyramidBlock2 = nn.Sequential(FPyramidBlock(self.out_features*4, self.out_features, self.dropout_prob))
        self.hidden_FPyramidBlock3 = nn.Sequential(FPyramidBlock(self.out_features*4, self.out_features, self.dropout_prob))
       
    def forward(self, x: torch.Tensor):
        output1 = self.hidden_FPyramidBlock1(x)
        output2 = self.hidden_FPyramidBlock2(output1)
        
        res_output2 = output1 + output2
        output3 = self.hidden_FPyramidBlock3(res_output2)
        return output3,res_output2
    
    def get_parameters(self) -> List[Dict]:
        return [{"params": self.parameters(), "lr": 1.}]
    
    def output_num(self):
        return self.out_features*4
    
class FusionFeatureFPN(nn.Module):
    def __init__(self, input_Rrs_dims:int, out_features:int=16, dropout_prob:float=0.1):
        super(FusionFeatureFPN, self).__init__()
        self.input_Rrs_dims = input_Rrs_dims 
        self.input_Rrsslope_dims = 6 
        self.out_features = out_features 
        self.dropout_prob = dropout_prob
        
        self.hidden_FPyramidBlock1 = nn.Sequential(FPyramidBlock(self.input_Rrs_dims-6, self.out_features, self.dropout_prob))
        
        self.linear1 = nn.Linear(self.input_Rrsslope_dims, self.out_features*4)
        self.bn1 = nn.BatchNorm1d(self.out_features*4)
        self.dp1 = nn.Dropout(p=self.dropout_prob)
        
        self.hidden_FPyramidBlock2 = nn.Sequential(FPyramidBlock(self.out_features*4*2, self.out_features, self.dropout_prob))
       
    def forward(self, x: torch.Tensor):
        x_spectral, x_slope = torch.split(x, [self.input_Rrs_dims-6, self.input_Rrsslope_dims], dim=1)
        
        output1 = self.hidden_FPyramidBlock1(x_spectral)
        output_slope = self.linear1(x_slope)
        output_slope = torch.relu(self.bn1(output_slope))
        output_slope = self.dp1(output_slope)
        
        shallow_output = output1 + output_slope 
        output2 = torch.cat((output1, output_slope), dim=1) 
        
        output3 = self.hidden_FPyramidBlock2(output2)
        return output3, shallow_output
    
    def get_parameters(self) -> List[Dict]:
        return [{"params": self.parameters(), "lr": 1.}]
    
    def output_num(self):
        return self.out_features*4
    
# In[4] Create TransferNet Model 
class TransferNet(nn.Module):
    def __init__(self, input_dims=13, hidden_dims=16, base_net='FFFPyramid', dropout_prob=0.1, 
                         transfer_loss='coral', use_bottleneck=True, bottleneck_width=10, width=64):
        super(TransferNet, self).__init__()
        self.base_network = network_dict[base_net](input_dims,hidden_dims,dropout_prob)
        self.use_bottleneck = use_bottleneck
        self.transfer_loss = transfer_loss
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.dropout_prob = dropout_prob
    
        bottleneck_list = [nn.Linear(self.base_network.output_num(
                    ), bottleneck_width), nn.BatchNorm1d(bottleneck_width), nn.ReLU(), nn.Dropout(self.dropout_prob)]
        self.bottleneck_layer = nn.Sequential(*bottleneck_list)
        regressor_layer_list = [nn.Linear(self.base_network.output_num(), width), nn.ReLU(), nn.Dropout(self.dropout_prob),
                                 nn.Linear(width, 1)]
        self.regressor_layer = nn.Sequential(*regressor_layer_list)

        self.bottleneck_layer[0].weight.data.normal_(0, 0.005)
        self.bottleneck_layer[0].bias.data.fill_(0.1)
        for i in range(2):
            self.regressor_layer[i * 3].weight.data.normal_(0, 0.01)
            self.regressor_layer[i * 3].bias.data.fill_(0.0)
        if self.transfer_loss == 'dann' or self.transfer_loss == 'adda':
            self.domain_discri = DomainDiscriminator(in_feature=self.hidden_dims*4, hidden_size=self.hidden_dims*4)
            self.use_bottleneck = False
            
    def forward(self, source, target, source_label):       
        source, source_shallow = self.base_network(source)
        target, target_shallow = self.base_network(target)
        source_reg = self.regressor_layer(source)
        target_reg = self.regressor_layer(target)
        if self.use_bottleneck:
            source = self.bottleneck_layer(source)
            target = self.bottleneck_layer(target)
            source_shallow = self.bottleneck_layer(source_shallow)
            target_shallow = self.bottleneck_layer(target_shallow)   
        kwargs = {}
        kwargs['source_label'] = source_label
        kwargs['target_logits'] = torch.nn.functional.softmax(target_reg, dim=1)
        transfer_lossr = self.adapt_loss(source, target, 
                                         source_shallow, target_shallow, self.transfer_loss, **kwargs)
        return source_reg, transfer_lossr

    def predict(self, x):
        features,_ = self.base_network(x)
        regssior_results = self.regressor_layer(features)
        return regssior_results
    
    def plot_sne(self, source, target): 
        source, source_shallow = self.base_network(source)
        target, target_shallow = self.base_network(target)
        if self.use_bottleneck:
            source = self.bottleneck_layer(source)
            target = self.bottleneck_layer(target)
            source_shallow = self.bottleneck_layer(source_shallow)
            target_shallow = self.bottleneck_layer(target_shallow) 
        return source, target

    def adapt_loss(self, X, Y, X_shallow, Y_shallow, adapt_loss, **kwargs):
        if adapt_loss == 'dan':
            mkmmd_loss = MultipleKernelMaximumMeanDiscrepancy(
                kernels = [GaussianKernel(alpha=2 ** k) for k in range(-3, 2)],linear=False)
            loss = mkmmd_loss(X, Y)
        elif adapt_loss == 'coral':
            CORAL = CorrelationAlignmentLoss()
            loss = CORAL(X, Y)
        elif adapt_loss == 'dann':
            dann = DomainAdversarialLoss(self.domain_discri, reduction='mean')
            loss = dann(X, Y)
        elif adapt_loss == 'jan':
            layer1_kernels = (GaussianKernel(alpha=0.5), GaussianKernel(1.), GaussianKernel(2.))
            layer2_kernels = (GaussianKernel(1.), )
            jan = JointMultipleKernelMaximumMeanDiscrepancy((layer1_kernels, layer2_kernels))
            loss = jan((X_shallow, X), (Y_shallow, Y))
        elif adapt_loss == 'adda':
            grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=2., max_iters=1000, auto_step=True)
            domain_adv = DomainAdversarialLoss(self.domain_discri, grl=grl)
            loss = domain_adv(X, Y)
        elif adapt_loss == 'gram':
            loss = DARE_GRAM_LOSS(X, Y)
        else:
            loss = 0
        return loss
    
    def get_optimizer(self, args):
        params = [
            {'params': self.base_network.parameters(), 'lr': 1 * args.lr},
            {'params': self.bottleneck_layer.parameters(), 'lr': 10 * args.lr},
            {'params': self.regressor_layer.parameters(), 'lr': 10 * args.lr},
        ]
        if self.transfer_loss == 'dann' or self.transfer_loss == 'adda':
            params.append({'params': self.domain_discri.parameters(), 'lr': 10 * args.lr})
        optimizer = torch.optim.SGD(
            params, lr=args.lr, momentum=args.momentum, weight_decay=args.lr_decay)
        return optimizer

# In[5] 创建消融实验相关的内容-消融Rrs_slope和金字塔架构 
class AblationBlock(nn.Module):
    def __init__(self, input_Rrs_dims:int, out_features:int=64, dropout_prob:float=0.1):
        super(AblationBlock, self).__init__()
        self.input_Rrs_dims = input_Rrs_dims
        self.out_features = out_features
        self.channel1 = FullConnectBlock(input_Rrs_dims, out_features, dropout_prob)
        self.channel2 = FullConnectBlock(out_features, out_features, dropout_prob)

    def forward(self, x):
        output1 = self.channel1(x)
        output2 = self.channel2(output1)
        return output2   
    
    def get_parameters(self) -> List[Dict]:
        return [{"params": self.parameters(), "lr": 1.}]
    
    def output_num(self):
        return self.out_features
    
class AblationRrsBlock(nn.Module):
    def __init__(self, input_Rrs_dims:int, out_features:int=64, dropout_prob:float=0.1):
        super(AblationRrsBlock, self).__init__()
        self.input_Rrs_dims = input_Rrs_dims 
        self.input_Rrsslope_dims = 6 
        self.out_features = out_features
        self.dropout_prob = dropout_prob
        
        self.hidden_AblationRrsBlock1 = FullConnectBlock(self.input_Rrs_dims-6, self.out_features, self.dropout_prob)
        
        self.linear1 = nn.Linear(self.input_Rrsslope_dims, self.out_features)
        self.bn1 = nn.BatchNorm1d(self.out_features)
        self.dp1 = nn.Dropout(p=self.dropout_prob)
        
        self.hidden_AblationRrsBlock2 = FullConnectBlock(self.out_features*2, self.out_features*2, self.dropout_prob)
       
    def forward(self, x: torch.Tensor):
        x_spectral, x_slope = torch.split(x, [self.input_Rrs_dims-6, self.input_Rrsslope_dims], dim=1)
        output1 = self.hidden_AblationRrsBlock1(x_spectral)
        output_slope = self.linear1(x_slope)
        output_slope = torch.relu(self.bn1(output_slope))
        output_slope = self.dp1(output_slope)

        output2 = torch.cat((output1, output_slope), dim=1) 
        output3 = self.hidden_AblationRrsBlock2(output2)
        
        return output3
    
    def get_parameters(self) -> List[Dict]:
        return [{"params": self.parameters(), "lr": 1.}]
    
    def output_num(self):
        return self.out_features*2

class AblationFPNBlock(nn.Module):
    def __init__(self, input_Rrs_dims:int, out_features:int=16, dropout_prob:float=0.1):
        super(AblationFPNBlock, self).__init__()
        self.input_Rrs_dims = input_Rrs_dims 
        self.out_features = out_features 
        self.dropout_prob = dropout_prob
        
        self.hidden_FPyramidBlock1 = nn.Sequential(FPyramidBlock(self.input_Rrs_dims, self.out_features, self.dropout_prob))
        self.hidden_FPyramidBlock2 = nn.Sequential(FPyramidBlock(self.out_features*4, self.out_features, self.dropout_prob))
       
    def forward(self, x: torch.Tensor):
        output1 = self.hidden_FPyramidBlock1(x)        
        output2 = self.hidden_FPyramidBlock2(output1)
        return output2
    
    def get_parameters(self) -> List[Dict]:
        return [{"params": self.parameters(), "lr": 1.}]
    
    def output_num(self):
        return self.out_features*4
       
class AblationNet(nn.Module):
    def __init__(self, input_dims=13, hidden_dims=16, base_net='FFFPyramid', dropout_prob=0.1, 
                         transfer_loss='coral', use_bottleneck=True, bottleneck_width=10, width=64):
        super(AblationNet, self).__init__()
        self.base_network = network_dict[base_net](input_dims, hidden_dims, dropout_prob)
        self.use_bottleneck = use_bottleneck
        self.transfer_loss = transfer_loss
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.dropout_prob = dropout_prob
    
        bottleneck_list = [nn.Linear(self.base_network.output_num(
                    ), bottleneck_width), nn.BatchNorm1d(bottleneck_width), nn.ReLU(), nn.Dropout(self.dropout_prob)]
        self.bottleneck_layer = nn.Sequential(*bottleneck_list)
        regressor_layer_list = [nn.Linear(self.base_network.output_num(), width), nn.ReLU(), nn.Dropout(self.dropout_prob),
                                 nn.Linear(width, 1)]
        self.regressor_layer = nn.Sequential(*regressor_layer_list)

        self.bottleneck_layer[0].weight.data.normal_(0, 0.005)
        self.bottleneck_layer[0].bias.data.fill_(0.1)
        for i in range(2):
            self.regressor_layer[i * 3].weight.data.normal_(0, 0.01)
            self.regressor_layer[i * 3].bias.data.fill_(0.0)
            
    def forward(self, source, target, source_label):       
        source = self.base_network(source)
        target = self.base_network(target)
        source_reg = self.regressor_layer(source)
        target_reg = self.regressor_layer(target)
        if self.use_bottleneck:
            source = self.bottleneck_layer(source)
            target = self.bottleneck_layer(target)   
        kwargs = {}
        kwargs['source_label'] = source_label
        kwargs['target_logits'] = torch.nn.functional.softmax(target_reg, dim=1)
        transfer_lossr = self.adapt_loss(source, target, self.transfer_loss, **kwargs)
        return source_reg, transfer_lossr

    def predict(self, x):
        features = self.base_network(x)
        regssior_results = self.regressor_layer(features)
        return regssior_results

    def adapt_loss(self, X, Y, adapt_loss, **kwargs):
        if adapt_loss == 'gram':
            loss = DARE_GRAM_LOSS(X, Y)
        else:
            loss = 0
        return loss
    
    def get_optimizer(self, args):
        params = [
            {'params': self.base_network.parameters(), 'lr': 1 * args.lr},
            {'params': self.bottleneck_layer.parameters(), 'lr': 10 * args.lr},
            {'params': self.regressor_layer.parameters(), 'lr': 10 * args.lr},
        ]
        optimizer = torch.optim.SGD(
            params, lr=args.lr, momentum=args.momentum, weight_decay=args.lr_decay)
        return optimizer    
        
network_dict = {"FPyramid": RegressorFPN,
                "FFFPyramid": FusionFeatureFPN,
                "AblationBlock": AblationBlock,
                "AblationRrsBlock": AblationRrsBlock,
                "AblationFPNBlock": AblationFPNBlock}
