if __name__ == "__main__":

    import os
    import sys

    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data.distributed import DistributedSampler
    from tqdm import tqdm

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # 内存碎片清理
    import torch
    from torch.utils.tensorboard import SummaryWriter

    from dataset.dataset_factory import get_dataset
    from models.genericloss import GenericLoss
    from models.model_tools import save_model, load_model
    from models.total import Total
    from utils.opts import opts

    opt = opts().parse()

    if opt.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus_str

    # 初始化分布式训练
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local_rank)

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
        pin_memory=True,
        drop_last=True,
    )

    # 模型
    model = Total(opt=opt).to(opt.device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), opt.lr, amsgrad=True)

    # 损失函数
    Loss = GenericLoss(opt=opt)

    # 训练
    global_step = 0
    start_epoch = 0
    num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
    loss_min = sys.maxsize  # 全局最小epoch损失

    if opt.resume:
        print("======加载恢复点======")
        model, start_epoch, optimizer, global_step, loss_min = load_model(model, opt.load_model, optimizer)

    # 开始训练
    for epoch in range(start_epoch, opt.num_epochs):
        epoch_loss_total = 0
        train_sampler.set_epoch(epoch)  # 设置epoch保证shuffle正确性
        if local_rank == 0:
            pbar = tqdm(data_loader, total=num_iters, desc=f"Epoch {epoch + 1}",
                        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
        else:
            pbar = data_loader

        for iter_id, batch in enumerate(pbar):
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

            rgb_branch, thermal_branch, output = model(
                batch['vi_image'],
                batch['ir_image'],
                pre_vi_img,
                pre_ir_img,
                pre_hm
            )

            # 计算损失
            loss_rgb, loss_stats_rgb = Loss(rgb_branch, batch)
            loss_thermal, loss_stats_thermal = Loss(thermal_branch, batch)
            loss_output, loss_stats_output = Loss(output, batch)
            total_loss = loss_rgb + loss_thermal + loss_output

            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            global_loss = total_loss.item() / dist.get_world_size()

            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # 只在主进程保存模型和记录日志
            if local_rank == 0:

                # TensorBoard日志记录
                if global_step % 100 == 0:
                    for name, value in loss_stats_rgb.items():
                        writer.add_scalar(f"RGBLoss/{name}", value.mean(), global_step)
                    for name, value in loss_stats_thermal.items():
                        writer.add_scalar(f"ThermalLoss/{name}", value.mean(), global_step)
                    for name, value in loss_stats_output.items():
                        writer.add_scalar(f"OutputLoss/{name}", value.mean(), global_step)
                    writer.add_scalar("Global_Loss", global_loss, global_step)
                    writer.add_scalar("Params/lr", optimizer.param_groups[0]['lr'], global_step)

                # 更新进度条信息
                pbar.set_postfix({
                    'loss': f"{global_loss:.4f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.2e}",
                    'step': global_step
                })
            epoch_loss_total += global_loss
            global_step += 1

        if local_rank == 0:
            epoch_loss_total /= len(data_loader)  # 计算这一个epoch的平均损失
            if epoch_loss_total < loss_min:
                loss_min = epoch_loss_total
                save_model(model=model.module,  # 注意获取原始模型
                           save_path=f'runs/best_model.pth',
                           epoch=epoch,
                           optimizer=optimizer, global_step=global_step, loss_min=loss_min)
                print(f"最佳模型为{epoch}，其损失为{loss_min}")
            save_model(model=model.module,  # 注意获取原始模型
                       save_path=f'runs/last.pth',
                       epoch=epoch,
                       optimizer=optimizer, global_step=global_step, loss_min=loss_min)
            pbar.close()  # 关闭当前epoch的进度条

        if epoch in opt.lr_step:
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    if local_rank == 0:
        writer.close()
        os.system('./test.sh')
    dist.destroy_process_group()
