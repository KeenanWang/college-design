import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from torch.utils.tensorboard import SummaryWriter

from dataset.dataset_factory import get_dataset
from models.genericloss import GenericLoss
from models.model_tools import save_model
from models.total import Total
from utils.opts import opts


def setup(rank, world_size):
    # 初始化进程组
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def main(rank, world_size, opt):
    # 初始化分布式训练
    setup(rank, world_size)

    # 初始化TensorBoard（仅主进程）
    if rank == 0:
        writer = SummaryWriter(log_dir=f'logs/exp_{opt.exp_id}')
    else:
        writer = None

    # 数据集
    torch.manual_seed(opt.seed + rank)  # 不同进程不同随机种子
    Dataset = get_dataset(opt.dataset)

    # 参数设置
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    if rank == 0:
        print(opt)

    # 分布式采样器
    train_dataset = Dataset(opt, 'train')
    sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    # 数据加载器
    data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        sampler=sampler,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )

    # 模型
    model = Total(opt=opt).to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)

    # 损失函数
    loss_func = GenericLoss(opt=opt)

    # AMP
    scaler = GradScaler(enabled=opt.use_amp)

    # 训练参数
    global_step = 0
    num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
    loss_min = sys.maxsize

    for epoch in range(opt.num_epochs):
        sampler.set_epoch(epoch)  # 重要：设置epoch保证shuffle正确

        if rank == 0:
            pbar = tqdm(total=num_iters, desc=f"Epoch {epoch + 1}",
                        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')

        model.train()
        for iter_id, batch in enumerate(data_loader):
            if iter_id >= num_iters:
                break

            # 数据迁移
            for k in batch:
                if k != 'meta':
                    batch[k] = batch[k].to(rank, non_blocking=True)

            # 前向传播
            with autocast(opt.use_amp):
                output = model(
                    batch['vi_image'],
                    batch['ir_image'],
                    batch.get('pre_vi_img', None),
                    batch.get('pre_ir_img', None),
                    batch.get('pre_hm', None)
                )

                # 计算损失
                loss, loss_stats = loss_func(output, batch)
                loss = loss.mean()

            # 反向传播
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 同步所有进程的损失值
            reduced_loss = loss.detach().clone()
            dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
            reduced_loss = reduced_loss / world_size

            # 主进程保存模型和日志
            if rank == 0:
                # 更新最小损失
                if reduced_loss < loss_min:
                    loss_min = reduced_loss.item()
                    save_model(model, 'runs/best_model.pth', epoch, optimizer)

                # 记录日志
                if global_step % 50 == 0:
                    writer.add_scalar("Loss/total", reduced_loss, global_step)
                    for name, value in loss_stats.items():
                        writer.add_scalar(f"Loss/{name}", value.mean(), global_step)
                    writer.add_scalar("Params/lr", optimizer.param_groups[0]['lr'], global_step)

                # 更新进度条
                pbar.update(1)
                pbar.set_postfix({
                    'loss': f"{reduced_loss.item():.4f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.2e}",
                    'step': global_step
                })

            global_step += 1
            torch.cuda.empty_cache()

        if rank == 0:
            pbar.close()

    if rank == 0:
        writer.close()
    cleanup()


if __name__ == "__main__":
    opt = opts().parse()

    # 设置可见GPU数量
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    world_size = len(opt.gpus)

    # 启动多进程训练
    mp.spawn(
        main,
        args=(world_size, opt),
        nprocs=world_size,
        join=True
    )