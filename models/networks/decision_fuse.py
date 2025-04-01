from torch import nn

from models.networks.single_unit_fusion import SingleUnitFusion


class DecisionFuse(nn.Module):
    """
    将三个分支的决策融合起来。
    """

    def __init__(self, c, h, w):
        super().__init__()
        # 全连接层
        self.hm = SingleUnitFusion(3, c, h, w)
        self.ltrb = SingleUnitFusion(3, 2 * c, h, w)
        self.reg = SingleUnitFusion(3, c, h, w)
        self.tracking = SingleUnitFusion(3, c, h, w)
        self.wh = SingleUnitFusion(3, c, h, w)

    def forward(self, rgb, thermal, fusion):
        hm_output = self.hm(rgb, thermal, fusion, 'hm')
        ltrb_output = self.ltrb(rgb, thermal, fusion, 'ltrb_amodal')
        reg_output = self.reg(rgb, thermal, fusion, 'reg')
        tracking_output = self.tracking(rgb, thermal, fusion, 'tracking')
        wh_output = self.wh(rgb, thermal, fusion, 'wh')
        return {
            'hm': hm_output,
            'ltrb_amodal': ltrb_output,
            'reg': reg_output,
            'tracking': tracking_output,
            'wh': wh_output,
        }
