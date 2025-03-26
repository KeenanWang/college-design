if __name__ == "__main__":

    import os
    import sys

    import torch.distributed as dist
    from torch.cuda.amp import GradScaler, autocast
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data.distributed import DistributedSampler
    from tqdm import tqdm

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # 内存碎片清理
    import torch
    from torch.utils.tensorboard import SummaryWriter

    from dataset.dataset_factory import get_dataset
    from models.genericloss import GenericLoss
    from models.model_tools import save_model
    from models.total import Total
    from utils.opts import opts

    # 初始化分布式训练
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local_rank)

    opt = opts().parse()

    # 只在主进程初始化TensorBoard
    if local_rank == 0:
        writer = SummaryWriter(log_dir=f'logs/exp_{opt.exp_id}')
    else:
        writer = None

    # 数据集
    torch.manual_seed(opt.seed + local_rank)  # 不同进程使用不同随机种子
    Dataset = get_dataset(opt.dataset)

    # 参数设置
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    opt.device = torch.device(f'cuda:{local_rank}')
    opt.batch_size = opt.batch_size // world_size  # 调整每个进程的batch size

    # 加载器
    train_dataset = Dataset(opt, 'train')
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=True
    )

    data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        sampler=train_sampler,
        num_workers=opt.num_workers,
        pin_memory=False,
        drop_last=True,
    )

    # 模型
    model = Total(opt=opt).to(opt.device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

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
        train_sampler.set_epoch(epoch)  # 设置epoch保证shuffle正确性
        if local_rank == 0:
            pbar = tqdm(data_loader, total=num_iters, desc=f"Epoch {epoch + 1}",
                        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
        else:
            pbar = data_loader

        for iter_id, batch in enumerate(pbar):
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
                loss = loss.mean()

            # 反向传播
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 只在主进程保存模型和记录日志
            if local_rank == 0:
                # 更新最小损失并保存模型
                if loss.item() < loss_min:
                    loss_min = loss.item()
                    save_model(model=model.module,  # 注意获取原始模型
                               save_path='runs/best_model.pth',
                               epoch=epoch,
                               optimizer=optimizer)

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

        if local_rank == 0:
            pbar.close()  # 关闭当前epoch的进度条

    if local_rank == 0:
        writer.close()
    dist.destroy_process_group()
