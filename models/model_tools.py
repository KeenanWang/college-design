import torch


def create_model(opt):
    from models.total import Total
    return Total(opt=opt)


def load_model(model, model_checkpoint):
    checkpoint = torch.load(model_checkpoint)
    model.load_state_dict(checkpoint['state_dict'])


def save_model(model, save_path, epoch, optimizer=None):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {'epoch': epoch,
            'state_dict': state_dict}
    if not (optimizer is None):
        data['optimizer'] = optimizer.state_dict()
    torch.save(data, save_path)
