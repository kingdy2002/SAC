import torch.nn as nn
import numpy as np
import torch
import torch.functional as F

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class conv2d(nn.Module) :
    def __init__(self,in_ch,out_ch,kernel_size,stride,padding):
        super(conv2d,self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,kernel_size=kernel_size,stride=stride,padding=padding,bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True)
        )
    def forward(self,x) :
        x = self.block(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3, padding=1):
        super(ResidualBlock, self).__init__()
        self.cnn = conv2d(in_ch,out_ch,kernel_size,stride,padding)
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        res = x
        x = self.cnn(x)
        x += self.shortcut(res)
        x = nn.ReLU(True)(x)
        return x


class Stem(nn.Module):
    def __init__(self,boardsize):
        super(Stem, self).__init__()

        self.boardsize = boardsize;
        self.block1 = nn.Sequential(
            conv2d(3,64,3,1,1),
            conv2d(64, 128, 3, 1,1)
        )
        self.block2 = nn.Sequential(
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128)
        )


    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x

class ValueNet(nn.Module) :
    def __init__(self,boardsize):
        super(ValueNet,self).__init__()
        self.boardsize = boardsize
        self.valueblock = nn.Sequential(
            conv2d(128, 4, 1, 1, 0),
            nn.Linear(4*self.boardsize*self.boardsize, self.boardsize*self.boardsize),
            nn.Tanh()
        )
    def forward(self, x):
        x = self.valueblock(x)
        return x

class PolicyNet(nn.Module) :
    def __init__(self, boardsize):
        super(PolicyNet,self).__init__()
        self.boardsize = boardsize
        self.policyblock = nn.Sequential(
            conv2d(128, 4, 1, 1, 0),
            nn.Linear(4 * self.boardsize * self.boardsize, self.boardsize * self.boardsize),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.policyblock(x)
        return x


class SAC(object):
    def __init__(self, boardsize) :
        self.boardsize = boardsize
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.targetentropy = -np.log((1.0 / (boardsize ** 2))) * 0.98
        self.stem = Stem(boardsize).to(self.device)
        self.local_ValueNet_1(boardsize).to(self.device)
        self.local_ValueNet_2(boardsize).to(self.device)
        self.PolicyNet(boardsize).to(self.device)
        self.alpha = torch.tensor(1).to(self.device)

        self.ValueNet_1(boardsize).to(self.device)
        self.ValueNet_2(boardsize).to(self.device)
        self.TargetNet(boardsize).to(self.device)


        self.actor_otim = torch.optim.adam(self.PolicyNet.parameters())

        self.value_otim_1 = torch.optim.adam(self.local_ValueNet_1.parameters())
        self.value_otim_2 = torch.optim.adam(self.local_ValueNet_2.parameters())

        self.alpha_otim = torch.optim.adam([[self.alpha]])
        hard_update(self.ValueNet_1,self.local_ValueNet_1)
        hard_update(self.ValueNet_2, self.local_ValueNet_2)
        hard_update(self.TargetNet, self.PolicyNet)

    def criticloss(self, state_batch, action_batch, reward_batch, next_state_batch, mask_batch):
        with torch.no_grad():
            after_stem_data = self.stem(next_state_batch)
            action_prob =  self.PolicyNet(after_stem_data)
            log_prob = torch.log(action_prob + 0.0001)
            q1 = self.ValueNet_1(after_stem_data)
            q2 = self.ValueNet_2(after_stem_data)
            min_q = action_prob*(torch.min(q1,q2) - self.alpha * log_prob)
            target_q = reward_batch + 0.99 + (1.0 - mask_batch) * min_q

        after_stem_data = self.stem(state_batch)
        qf1 = self.local_ValueNet_1(after_stem_data).gather(1, action_batch.long())
        qf2 = self.local_ValueNet_2(after_stem_data).gather(1, action_batch.long())
        qf1_loss = F.mse_loss(qf1, target_q)
        qf2_loss = F.mse_loss(qf2, target_q)
        return qf1_loss, qf2_loss

    def policyloss(self, state_batch):
        after_stem_data = self.stem(state_batch)
        q1 = self.local_ValueNet_1(after_stem_data)
        q2 = self.local_ValueNet_2(after_stem_data)
        q = torch.min(q1,q2)
        action_prob = self.PolicyNet(after_stem_data)
        loss = (self.alpha * action_prob - q)*action_prob
        return loss.sum(dim=1).mean()


    def updatecritic(self,loss,tau):
        self.value_otim_1.zero_grad()
        loss.backward()
        self.value_otim_1.step()

        self.local_ValueNet_2.zero_grad()
        loss.backward()
        self.local_ValueNet_2.step()

        soft_update(self.ValueNet_1,self.local_ValueNet_1,tau)
        soft_update(self.ValueNet_2,self.local_ValueNet_2,tau)

    def updatepolicy(self,loss,tau):

        self.actor_otim.zero_grad()
        loss.backward()
        self.actor_otim.step()
        soft_update(self.TargetNet,self.PolicyNet,tau)

    def updatealpha(self,loss,tau):

        self.actor_otim.zero_grad()
        loss.backward()
        self.actor_otim.step()
        soft_update(self.TargetNet,self.PolicyNet,tau)

        ##알파 로스 랑 알파 업데이트 아직 안만듬