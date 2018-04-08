#encoding:utf8

from torch import nn
from torch.nn import Parameter
import torch as T
from torch.autograd import Variable
import torch.nn.functional as F
import math




class IndRNNCell(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='relu', hidden_min_abs=0, hidden_max_abs=None):
        super(IndRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        self.hidden_min_abs = hidden_min_abs
        self.hidden_max_abs = hidden_max_abs
        self.bias = bias
        self.weight_ih = Parameter(T.Tensor(hidden_size, input_size))
        self.weight_hh = Parameter(T.Tensor(hidden_size))

        if bias:
            self.bias_ih = Parameter(T.Tensor(hidden_size))

        else:
            self.register_parameter('bias_ih', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)

        for name, weight in self.named_parameters():
            if 'bias' in name:
                weight.data.zero_()
            elif "weight_hh" in name:
                if self.hidden_max_abs:
                    stdv_ = self.hidden_max_abs
                else:
                    stdv_ = stdv

                weight.data.uniform_(-stdv_, stdv_)
            elif "weight_ih" in name:
                weight.data.normal_(0, 0.01)
            else:
                weight.data.normal_(0, 0.01)

        self.check_bounds()

    def check_bounds(self):

        if self.hidden_min_abs:
            abs_kernel = T.abs(self.weight_hh.data)
            min_abs_kernel = T.clamp(abs_kernel, min=self.hidden_min_abs)

            self.weight_hh.data.copy_(
                T.mul(T.sign(self.weight_hh.data), min_abs_kernel)
            )

        if self.hidden_max_abs:
            self.weight_hh.data.copy_(
                T.clamp(self.weight_hh.data, max=self.hidden_max_abs, min=-self.hidden_max_abs)
            )

    def IndRNNTanhCell(self, input, hidden, w_ih, w_hh, b_ih=None):
        hy = F.tanh(F.linear(input, w_ih, b_ih) + F.mul(w_hh, hidden))
        return hy

    def IndRNNReluCell(self, input, hidden, w_ih, w_hh, b_ih=None):
        hy = F.relu(F.linear(input, w_ih, b_ih) + F.mul(w_hh, hidden))
        return hy

    def forward(self, input, hx):
        if self.nonlinearity == 'tanh':
            func = self.IndRNNTanhCell
        elif self.nonlinearity == 'relu':
            func = self.IndRNNReluCell

        return func(input, hx, self.weight_ih, self.weight_hh, self.bias_ih)


class IndRNN(nn.Module):

    def __init__(self, input_size, hidden_size, n_layer=1, batch_norm=False,
                 step_size=None, **kwargs):
        super(IndRNN, self).__init__()
        self.hidden_size = hidden_size
        if batch_norm and step_size is None:
            raise Exception("Frame wise batch size needs to know the step size")
        self.batch_norm = batch_norm
        self.step_size = step_size
        self.n_layer = n_layer

        cells = []
        for i in range(n_layer):
            if i == 0:
                cells += [IndRNNCell(input_size, hidden_size, **kwargs)]
            else:
                cells += [IndRNNCell(hidden_size, hidden_size, **kwargs)]
        self.cells = nn.ModuleList(cells)

        if batch_norm:
            bns = []
            for i in range(n_layer):
                bns += [nn.BatchNorm2d(step_size)]
            self.bns = nn.ModuleList(bns)



    def forward(self, x, hx=None):
        '''

        :param x: (time_step,batch_size,dim)
        :param hx:
        :return:
        '''
        hidden = []
        if hx is None:
            hx = T.autograd.Variable(x.data.new(self.n_layer,x.size(1),
                                                        self.hidden_size).zero_(), requires_grad=False).contiguous()
        for i, cell in enumerate(self.cells):
            hx = hx[i]
            cell.check_bounds()
            outputs = []
            for t in range(x.size(0)):
                x_t = x[t]
                hx = cell(x_t, hx)
                outputs += [hx]
            hidden.append(hx)

            x = T.stack(outputs, 0)
            if self.batch_norm:
                T.backends.cudnn.enabled = False
                x = self.bns[i](x.permute(1,0,2).contiguous()).permute(1,0,2).cuda()

        hidden = T.stack(hidden,0)
        return x, hidden



if __name__ == "__main__":
    rnn = IndRNN(10, 20, 2)
    input = Variable(T.randn(5, 3, 10))
    h0 = Variable(T.randn(2, 3, 20))
    output,_= rnn(input, h0)
    raw_rnn = nn.RNN(10,20,2)
    output_raw,hidden = raw_rnn(input,h0)
    print(output)