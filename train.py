import os
import sys

from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm  # 新增导入

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # 内存碎片清理
import torch
from torch.utils.tensorboard import SummaryWriter

from dataset.dataset_factory import get_dataset
from models.genericloss import GenericLoss
from models.model_tools import save_model
from models.total import Total
from utils.opts import opts

opt = opts().parse()

# 初始化TensorBoard
writer = SummaryWriter(log_dir=f'logs/exp_{opt.exp_id}')

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

# AMP
scaler = GradScaler(enabled=opt.use_amp)

# 训练
global_step = 0
num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
loss_min = sys.maxsize

for epoch in range(opt.num_epochs):
    print(f"\nEpoch {epoch + 1}/{opt.num_epochs}")

    # 创建进度条
    pbar = tqdm(data_loader, total=num_iters, desc=f"Epoch {epoch + 1}",
                bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')

    for iter_id, batch in enumerate(pbar):  # 修改为迭代pbar
        if iter_id >= num_iters:
            break

        torch.cuda.empty_cache()

        # 数据迁移到设备
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].to(device=opt.device, non_blocking=True)

        # 前向传播
        pre_vi_img = batch.get('pre_vi_img', None)
        pre_ir_img = batch.get('pre_ir_img', None)
        pre_hm = batch.get('pre_hm', None)

        with autocast(opt.use_amp):
            output = model(
                batch['vi_image'],
                batch['ir_image'],
                pre_vi_img,
                pre_ir_img,
                pre_hm
            )

            # 计算损失
            loss, loss_stats = Loss(output, batch)
            loss = loss.mean() / opt.batch_size

        # 反向传播
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # 更新最小损失并保存模型
        if loss.item() < loss_min:
            loss_min = loss.item()
            save_model(model=model, save_path='runs/best_model.pth', epoch=epoch, optimizer=optimizer)

        # TensorBoard日志记录
        if global_step % 50 == 0:
            for name, value in loss_stats.items():
                writer.add_scalar(f"Loss/{name}", value.mean(), global_step)
            writer.add_scalar("Params/lr", optimizer.param_groups[0]['lr'], global_step)

        # 更新进度条信息
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.2e}",
            'step': global_step
        })

        global_step += 1
        del output, loss, loss_stats

    pbar.close()  # 关闭当前epoch的进度条

writer.close()
