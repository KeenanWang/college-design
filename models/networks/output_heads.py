from torch import nn


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class OutputHeads(nn.Module):
    def __init__(self, heads, head_convs, num_stacks, last_channel, opt=None):
        super().__init__()
        self.opt = opt
        if opt is not None and opt.head_kernel != 3:
            print('Using head kernel:', opt.head_kernel)
            head_kernel = opt.head_kernel
        else:
            head_kernel = 3
        self.num_stacks = num_stacks
        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            head_conv = head_convs[head]
            if len(head_conv) > 0:
                out = nn.Conv2d(head_conv[-1], classes,
                                kernel_size=1, stride=1, padding=0, bias=True)
                conv = nn.Conv2d(last_channel, head_conv[0],
                                 kernel_size=head_kernel,
                                 padding=head_kernel // 2, bias=True)
                convs = [conv]
                for k in range(1, len(head_conv)):
                    convs.append(nn.Conv2d(head_conv[k - 1], head_conv[k],
                                           kernel_size=1, bias=True))
                if len(convs) == 1:
                    fc = nn.Sequential(conv, nn.ReLU(inplace=True), out)
                elif len(convs) == 2:
                    fc = nn.Sequential(
                        convs[0], nn.ReLU(inplace=True),
                        convs[1], nn.ReLU(inplace=True), out)
                elif len(convs) == 3:
                    fc = nn.Sequential(
                        convs[0], nn.ReLU(inplace=True),
                        convs[1], nn.ReLU(inplace=True),
                        convs[2], nn.ReLU(inplace=True), out)
                elif len(convs) == 4:
                    fc = nn.Sequential(
                        convs[0], nn.ReLU(inplace=True),
                        convs[1], nn.ReLU(inplace=True),
                        convs[2], nn.ReLU(inplace=True),
                        convs[3], nn.ReLU(inplace=True), out)
                if 'hm' in head:
                    fc[-1].bias.data.fill_(opt.prior_bias)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(last_channel, classes,
                               kernel_size=1, stride=1, padding=0, bias=True)
                if 'hm' in head:
                    fc.bias.data.fill_(opt.prior_bias)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def forward(self, feats):
        out = []
        if self.opt.model_output_list:
            for s in range(self.num_stacks):
                z = []
                for head in sorted(self.heads):
                    z.append(self.__getattr__(head)(feats[s]))
                out.append(z)
        else:
            for s in range(self.num_stacks):
                z = {}
                for head in self.heads:
                    z[head] = self.__getattr__(head)(feats[s])
                out.append(z)
        return out


if __name__ == '__main__':
    import torch
    from utils.opts import opts

    opt = opts().init()
    feats = [torch.randn(4, 64, 136, 240)]
    model = OutputHeads(heads={'hm': 2, 'ltrb_amodal': 4, 'reg': 2, 'tracking': 2, 'wh': 2},
                        head_convs={'hm': [256], 'ltrb_amodal': [256], 'reg': [256], 'tracking': [256],
                                    'wh': [256]},
                        num_stacks=1, last_channel=64, opt=opt)
    output = model(feats)[-1]
    for each in output:
        print(each, output[each].shape)
