import os
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # 内存碎片清理
import torch
from torch.utils.tensorboard import SummaryWriter  # 新增导入
import torchvision.utils as vutils  # 新增导入

from dataset.dataset_factory import get_dataset
from models.genericloss import GenericLoss
from models.model_tools import save_model
from models.total import Total
from utils.opts import opts

opt = opts().parse()

# 初始化TensorBoard
writer = SummaryWriter(log_dir=f'logs/exp_{opt.exp_id}')  # 新增代码

# 数据集
torch.manual_seed(opt.seed)
torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
Dataset = get_dataset(opt.dataset)

# 参数设置
opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
print(opt)
if not opt.not_set_cuda_env:
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

# 加载器
data_loader = torch.utils.data.DataLoader(
    Dataset(opt, 'train'),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.num_workers,
    pin_memory=False,
    drop_last=True
)

# 模型
model = Total(opt=opt)
model = model.to(opt.device)

# 优化器
optimizer = torch.optim.Adam(model.parameters(), opt.lr)

# 损失函数
Loss = GenericLoss(opt=opt)

# 训练
global_step = 0  # 新增全局步数计数器
num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
loss_min = sys.maxsize

for epoch in range(opt.num_epochs):
    for iter_id, batch in enumerate(data_loader):
        torch.cuda.empty_cache()  # 清除未使用的显存
        if iter_id >= num_iters:
            break
        # 数据迁移到设备
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].to(device=opt.device, non_blocking=True)

        # 前向传播
        pre_vi_img = batch.get('pre_vi_img', None)
        pre_ir_img = batch.get('pre_ir_img', None)
        pre_hm = batch.get('pre_hm', None)

        output = model(
            batch['vi_image'],
            batch['ir_image'],
            pre_vi_img,
            pre_ir_img,
            pre_hm
        )

        # 计算损失
        loss, loss_stats = Loss(output, batch)
        loss = loss.mean()

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss.item() < loss_min:
            loss_min = loss.item()
            save_model(model=model, save_path='runs/best_model.pth', epoch=epoch, optimizer=optimizer)

        # TensorBoard日志记录 新增代码块
        if global_step % 100 == 0:  # 每100步记录一次标量
            # 记录损失
            for name, value in loss_stats.items():
                writer.add_scalar(f"Loss/{name}", value.mean(), global_step)

            # 记录学习率
            writer.add_scalar("Params/lr", optimizer.param_groups[0]['lr'], global_step)

        if global_step % 500 == 0:  # 每500步记录一次图像
            with torch.no_grad():
                # 可视化输入和输出图像（假设输出包含融合图像）
                vi_grid = vutils.make_grid(batch['vi_image'][:4], normalize=True)
                ir_grid = vutils.make_grid(batch['ir_image'][:4], normalize=True)
                fusion_grid = vutils.make_grid(output['fusion'][:4], normalize=True)  # 需要确认输出键名

                writer.add_image('Images/Visible', vi_grid, global_step)
                writer.add_image('Images/Infrared', ir_grid, global_step)
                writer.add_image('Images/Fusion', fusion_grid, global_step)

        global_step += 1  # 更新全局步数

        del output, loss, loss_stats

writer.close()  # 关闭记录器
