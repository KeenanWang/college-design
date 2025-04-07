import torch
from torch.nn.parallel import DistributedDataParallel as DDP


def create_model(opt):
    from models.total import Total
    return Total(opt=opt)


def load_model(model,
               model_path,
               optimizer=None,
               scaler=None,
               strict=False,
               verbose=True):
    """
    改进版模型加载函数，支持分布式训练和完整训练状态恢复

    参数:
    model:       要加载的模型 (常规模型或DDP包装的模型)
    model_path:  检查点路径
    optimizer:   优化器对象 (可选)
    scaler:      GradScaler对象 (可选)
    strict:      是否严格加载模型参数 (默认False)
    verbose:     是否显示加载详情 (默认True)

    返回:
    (model, epoch, optimizer, scaler, global_step, loss_min)
    """

    # 自动处理设备映射
    map_location = lambda storage, loc: storage.cuda() if torch.cuda.is_available() else storage

    # 加载检查点
    checkpoint = torch.load(model_path, map_location=map_location)
    state_dict = checkpoint['state_dict']

    # 处理DDP包装的模型
    if isinstance(model, DDP):
        model.module.load_state_dict(state_dict, strict=strict)
    else:
        model.load_state_dict(state_dict, strict=strict)

    if optimizer is not None:
        # 加载优化器状态
        optimizer.load_state_dict(checkpoint['optimizer'])

        # 加载混合精度状态
        scaler.load_state_dict(checkpoint['scaler'])

        # 获取训练状态信息
        epoch = checkpoint.get('epoch', 0)
        global_step = checkpoint.get('global_step', 0)
        loss_min = checkpoint.get('loss_min', float('inf'))

        # 打印加载信息
        if verbose:
            print(f"\n成功加载检查点: {model_path}")
            print(f"  恢复训练起始周期: {epoch + 2} (已训练{epoch + 1}个周期)")
            print(f"  全局训练步数: {global_step}")
            print(f"  历史最小损失: {loss_min:.4f}")
        return model, epoch + 1, optimizer, scaler, global_step, loss_min

    else:
        # 一般加载函数
        return model


def save_model(model, save_path, epoch, optimizer=None, scaler=None, global_step=None, loss_min=None):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {'epoch': epoch,
            'state_dict': state_dict, }
    if optimizer is not None:
        data['optimizer'] = optimizer.state_dict()
        data['scaler'] = scaler.state_dict()
        data['global_step'] = global_step
        data['loss_min'] = loss_min
    torch.save(data, save_path)
