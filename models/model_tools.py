import torch


def create_model(opt):
    from models.total import Total
    return Total(opt=opt)


def load_model(model, model_path):
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    return model


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
