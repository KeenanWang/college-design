import torch

from models.Losses.losses import FastFocalLoss, RegWeightedL1Loss
from models.Losses.utils import _sigmoid


class GenericLoss(torch.nn.Module):
    def __init__(self, opt):
        super(GenericLoss, self).__init__()
        self.crit = FastFocalLoss()
        self.crit_reg = RegWeightedL1Loss()
        self.opt = opt

    def _sigmoid_output(self, output):
        if 'hm' in output:
            output['hm'] = _sigmoid(output['hm'])
        return output

    def forward(self, outputs, batch):
        opt = self.opt
        losses = {head: 0 for head in opt.heads}

        for s in range(opt.num_stacks):
            output = outputs[s]
            output = self._sigmoid_output(output)

            if 'hm' in output:
                losses['hm'] += self.crit(
                    output['hm'], batch['hm'], batch['ind'],
                    batch['mask'], batch['cat']) / opt.num_stacks

            regression_heads = [
                'reg', 'wh', 'tracking', 'ltrb', 'ltrb_amodal', 'hps',
                'dep', 'dim', 'amodel_offset', 'velocity']

            for head in regression_heads:
                if head in output:
                    losses[head] += self.crit_reg(
                        output[head], batch[head + '_mask'],
                        batch['ind'], batch[head]) / opt.num_stacks

        losses['tot'] = 0
        for head in opt.heads:
            losses['tot'] += opt.weights[head] * losses[head]

        return losses['tot'], losses
