#encoding:utf8

from IndRNN import *
import torch as T
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets,transforms
import numpy as np

import argparse

parser = argparse.ArgumentParser(description='PyTorch IndRNN Addition test')
# Default parameters taken from https://arxiv.org/abs/1803.04831
parser.add_argument('--lr', type=float, default=0.0002,
                    help='learning rate (default: 0.0002)')
parser.add_argument('--n-layer', type=int, default=6,
                    help='number of layer of IndRNN (default: 6)')
parser.add_argument('--hidden_size', type=int, default=128,
                    help='number of hidden units in one IndRNN layer(default: 128)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--no-batch-norm', action='store_true', default=False,
                    help='disable frame-wise batch normalization after each layer')
parser.add_argument('--log_epoch', type=int, default=1,
                    help='after how many iterations to report performance')


parser.add_argument('--batch-size', type=int, default=256,
                    help='input batch size for training (default: 256)')
parser.add_argument('--max-steps', type=int, default=10000,
                    help='input batch size for training (default: 10000)')
args = parser.parse_args()
args.cuda = not args.no_cuda and T.cuda.is_available()
args.batch_norm = not args.no_batch_norm

# Parameters taken from https://arxiv.org/abs/1803.04831
TIME_STEPS = 784 # 28x28 pixels
RECURRENT_MAX = pow(2, 1 / TIME_STEPS)


cuda = T.cuda.is_available()


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, n_layer=2):
        super(Net, self).__init__()
        self.indrnn = IndRNN(
            input_size, hidden_size, n_layer, batch_norm=args.batch_norm,
            hidden_max_abs=RECURRENT_MAX, step_size=TIME_STEPS)
        self.lin = nn.Linear(hidden_size, 10)
        self.lin.bias.data.fill_(.1)
        self.lin.weight.data.normal_(0, .01)

    def forward(self, x, hidden=None):
        y,_ = self.indrnn(x, hidden)
        return self.lin(y[-1]).squeeze(1)



def main():
    # build model
    model = Net(1, args.hidden_size, args.n_layer)
    # model = LSTM()
    if cuda:
        model.cuda()
    optimizer = T.optim.Adam(model.parameters(), lr=args.lr)

    # load data
    train_data, test_data = sequential_MNIST(args.batch_size, cuda=cuda)

    # Train the model
    model.train()
    step = 0
    epochs = 0
    while step < args.max_steps:
        losses = []
        for data, target in train_data:
            data, target = Variable(data), Variable(target)
            if cuda:
                data, target = data.cuda(), target.cuda()

            data = data.permute(1,0,2)
            model.zero_grad()
            out = model(data)
            loss = F.cross_entropy(out, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.data.cpu()[0])
            step += 1
            if step >= args.max_steps:
                break
        if epochs % args.log_epoch == 0:
            print(
                "Epoch {} cross_entropy {}".format(
                    epochs, np.mean(losses)))
        epochs += 1

    # get test error
    model.eval()
    correct = 0
    for data, target in test_data:
        data, target = Variable(data), Variable(target)
        if cuda:
            data, target = data.cuda(), target.cuda()

        data = data.permute(1,0,2)
        out = model(data)
        pred = out.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    print(
        "Test accuracy:: {:.4f}".format(
            100. * correct / len(test_data.dataset)))




def sequential_MNIST(batch_size, cuda=False, dataset_folder='./data'):
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    train_loader = T.utils.data.DataLoader(
        datasets.MNIST(dataset_folder, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,)),
                           # transform to sequence
                           transforms.Lambda(lambda x: x.view(-1, 1))
                       ])),
        batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)

    test_loader = T.utils.data.DataLoader(
        datasets.MNIST(dataset_folder, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            # transform to sequence
            transforms.Lambda(lambda x: x.view(-1, 1))
        ])),
        batch_size=batch_size, shuffle=False, **kwargs)

    return (train_loader, test_loader)


if __name__ == "__main__":
    main()













