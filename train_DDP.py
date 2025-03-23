import os
import sys

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler  # 📌 新增导入

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

from dataset.dataset_factory import get_dataset
from models.genericloss import GenericLoss
from models.model_tools import save_model
from models.total import Total
from utils.opts import opts


# 📌 初始化分布式环境
def init_distributed_mode(opt):
    opt.rank = int(os.environ['RANK'])
    opt.local_rank = int(os.environ['LOCAL_RANK'])
    opt.world_size = int(os.environ['WORLD_SIZE'])

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=opt.world_size,
        rank=opt.rank
    )

    # 确保每个进程使用不同的GPU
    torch.cuda.set_device(opt.local_rank)
    opt.device = torch.device("cuda", opt.local_rank)

    # 同步所有进程
    dist.barrier()


def main():
    opt = opts().parse()

    # 📌 初始化分布式训练
    if opt.distributed:  # 添加新的命令行参数 --distributed
        init_distributed_mode(opt)
        print(f"Initialized distributed training on rank {opt.rank}")
    else:
        opt.rank = 0  # 单卡模式默认rank为0
        opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    # 📌 只在主进程初始化TensorBoard
    if opt.rank == 0:
        writer = SummaryWriter(log_dir=f'logs/exp_{opt.exp_id}')
    else:
        writer = None

    # 数据集
    torch.manual_seed(opt.seed + opt.rank)  # 📌 不同rank设置不同随机种子
    Dataset = get_dataset(opt.dataset)
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)

    # 📌 注释掉手动设置CUDA_VISIBLE_DEVICES（由启动脚本控制）
    # if not opt.not_set_cuda_env:
    #     os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

    # 📌 分布式采样器
    train_dataset = Dataset(opt, 'train')
    sampler = DistributedSampler(
        train_dataset,
        num_replicas=opt.world_size if opt.distributed else 1,
        rank=opt.rank,
        shuffle=not opt.not_shuffle  # 📌 采样器控制shuffle
    ) if opt.distributed else None

    data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=(sampler is None),
        num_workers=opt.num_workers,
        pin_memory=True,  # 📌 建议开启加速数据传输
        sampler=sampler,
        drop_last=True
    )

    # 模型
    model = Total(opt=opt)
    model = model.to(opt.device)

    # 📌 DDP包装模型
    if opt.distributed:
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank)

    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    Loss = GenericLoss(opt=opt)

    # 训练
    global_step = 0
    num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
    loss_min = sys.maxsize

    for epoch in range(opt.num_epochs):
        # 📌 设置epoch为sampler的随机种子
        if opt.distributed:
            data_loader.sampler.set_epoch(epoch)

        for iter_id, batch in enumerate(data_loader):
            if iter_id >= num_iters:
                break

            # 📌 清除显存改为按需执行
            if iter_id % 10 == 0:
                torch.cuda.empty_cache()

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

            # 📌 只在主进程保存模型和记录日志
            if opt.rank == 0:
                if loss.item() < loss_min:
                    loss_min = loss.item()
                    save_model(
                        model=model.module if opt.distributed else model,  # 📌 注意获取原始模型
                        save_path='runs/best_model.pth',
                        epoch=epoch,
                        optimizer=optimizer
                    )

                # TensorBoard日志记录
                if global_step % 100 == 0:
                    for name, value in loss_stats.items():
                        writer.add_scalar(f"Loss/{name}", value.mean(), global_step)
                    writer.add_scalar("Params/lr", optimizer.param_groups[0]['lr'], global_step)

                if global_step % 500 == 0:
                    with torch.no_grad():
                        vi_grid = vutils.make_grid(batch['vi_image'][:4], normalize=True)
                        ir_grid = vutils.make_grid(batch['ir_image'][:4], normalize=True)
                        fusion_grid = vutils.make_grid(output['fusion'][:4], normalize=True)
                        writer.add_image('Images/Visible', vi_grid, global_step)
                        writer.add_image('Images/Infrared', ir_grid, global_step)
                        writer.add_image('Images/Fusion', fusion_grid, global_step)

            global_step += 1
            del output, loss, loss_stats

    if opt.rank == 0 and writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
