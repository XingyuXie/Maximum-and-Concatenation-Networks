import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms as transforms
import numpy as np
import pdb
import argparse
import torch
import random
# fix random seeds
torch.manual_seed(1234)
#torch.cuda.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
np.random.seed(1234)
random.seed(1234)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


from models import *
from misc import progress_bar

import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"


CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def main():
    parser = argparse.ArgumentParser(description="cifar-10 with PyTorch")
    # parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
    #parser.add_argument('--lr_MCN', default=0.01, type=float, help='learning rate for MCN')
    # parser.add_argument('--epoch', default=300, type=int, help='number of epochs tp train for')
    parser.add_argument('--epoch', default=290, type=int, help='number of epochs tp train for')
    parser.add_argument('--trainBatchSize', default=100, type=int, help='training batch size')
    parser.add_argument('--testBatchSize', default=100, type=int, help='testing batch size')
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='whether cuda is in use')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--decay', type=float, default=7e-4, help='Weight decay (L2 penalty).')
    args = parser.parse_args()

    solver = Solver(args)
    solver.run()


class Solver(object):
    def __init__(self, config):
        self.model = None
        self.lr = config.lr
        #self.lr_MCN = config.lr_MCN
        self.momentum = config.momentum
        self.weight_decay = config.decay
        
        
        self.epochs = config.epoch
        self.train_batch_size = config.trainBatchSize
        self.test_batch_size = config.testBatchSize
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.device = None
        self.cuda = config.cuda
        self.train_loader = None
        self.test_loader = None
        self.MCNparam = list()
        self.other_param = list()

    def load_data(self):
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
        self.train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=self.train_batch_size, shuffle=True)
        test_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=self.test_batch_size, shuffle=False)

    def load_model(self):
        if self.cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # self.model = LeNet().to(self.device)
        # self.model = LM_CVNN_13().to(self.device)
        #self.model = LM_CVNN_12().to(self.device)
        #self.model = LM_CVNN_11().to(self.device)
        #self.model = LM_CVNN_10().to(self.device)
        #        self.model = LM_CVNN_9().to(self.device)



        # self.model = VGG11().to(self.device)
        # self.model = VGG13().to(self.device)
        # self.model = VGG16().to(self.device)
        # self.model = VGG19().to(self.device)
        # self.model = GoogLeNet().to(self.device)
        # self.model = resnet18_CNN().to(self.device)
        
        
        # self.model = resnext50_32x4d(pretrained = True, num_classes = 1000)
        # self.model = resnext50_32x4d(pretrained = True)
        # self.model.fc = nn.Linear(in_features=2048, out_features=100, bias=True)
        
        
        self.model = CifarResNeXt_MCN(8, 29, 100, 64, 4)
        print(self.model)


        self.MCNparam = [param for param in self.model.MCN_block1.parameters() if param.requires_grad]
        self.MCNparam += [param for param in self.model.MCN_block2.parameters() if param.requires_grad]
        self.MCNparam += [param for param in self.model.classifier.parameters() if param.requires_grad]
        print(len(self.MCNparam))
        # pdb.set_trace()
        
        filter_list = [id(param) for param in self.MCNparam]
        self.other_param = [param for param in filter(lambda p: id(p) not in filter_list,self.model.parameters())]
        
        print(len(self.other_param))
        # pdb.set_trace()
        



        #pre_model = torch.load('model_ResNeXt_cnn.pth', map_location='cpu') # model
        # torch.save(pre_model.cpu().state_dict(), 'RNX_Pre.pth')

        # pretrained_net = torch.load('RNX_Pre.pth')  # state_dict
        # self.model.load_state_dict(pretrained_net, strict=False)
        #
        # pdb.set_trace()
        #
        # for key, v in enumerate(pretrained_net):
        #     exec(r"self.model." + v + r".requires_grad = False")
        #     #
        # for key, v in enumerate(pretrained_net):
        #     print(key, eval(r"self.model." + v + r".requires_grad"))
        


        
        


        self.model = self.model.to(self.device)

        #self.optimizer = torch.optim.SGD([{'params': self.other_param, 'lr': self.lr},{'params': self.MCNparam, 'lr': self.lr_MCN},], 
       #momentum = self.momentum ,weight_decay=self.weight_decay, nesterov=True)
        #self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum = self.momentum ,weight_decay=self.weight_decay, nesterov=True)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=np.linspace(20,280,15,dtype=np.int), gamma=0.75)
        # self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[150, 225], gamma=0.1)
        # self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[150, 225], gamma=0.1)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def train(self):
        print("train:")
        self.model.train()
        train_loss = 0
        train_correct = 0
        total = 0

        for batch_num, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            # pdb.set_trace()
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced
            total += target.size(0)

            # train_correct incremented by one if predicted right
            
            train_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

            if (batch_num + 1) % 100 == 0:
                progress_bar(batch_num, len(self.train_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                             % (train_loss / (batch_num + 1), 100. * train_correct / total, train_correct, total))

        return train_loss, train_correct / total

    def test(self):
        print("test:")
        self.model.eval()
        test_loss = 0
        test_correct = 0
        total = 0

        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                test_loss += loss.item()
                prediction = torch.max(output, 1)
                total += target.size(0)
                test_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

                if (batch_num + 1) % 50 ==0 :
                    progress_bar(batch_num, len(self.test_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                                 % (test_loss / (batch_num + 1), 100. * test_correct / total, test_correct, total))

        return test_loss, test_correct / total

    def save(self):
        model_out_path = "model_ResNeXt_cnn.pth"
        torch.save(self.model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def run(self):
        self.load_data()
        self.load_model()
        accuracy = 0
        train_loss  = 10000
        train_acc  = 0

        for epoch in range(1, self.epochs + 1):
            self.scheduler.step(epoch)
            print("\n===> epoch: %d/300" % epoch)
            train_result = self.train()
            print(train_result)
            test_result = self.test()
            accuracy = max(accuracy, test_result[1])
            train_loss = min(train_loss, train_result[0])
            train_acc = max(train_acc, train_result[1])
            print("===> BEST ACC. PERFORMANCE: %.3f%%" % (accuracy * 100))
            print("===> BEST TRAIN LOSS PERFORMANCE: %.3f%%" % (train_loss/500))
            print("===> BEST TRAIN ACC PERFORMANCE: %.3f%%" % (train_acc * 100))


            
            if epoch == self.epochs:
                print("===> BEST ACC. PERFORMANCE: %.3f%%" % (accuracy * 100))
                # self.save()



if __name__ == '__main__':
    main()
