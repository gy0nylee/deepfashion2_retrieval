import torchvision.models as models

def get_model(args):
    if args.model == 'resnet18':
        return models.resnet18()
    if args.model == 'resnet34':
        return models.resnet34()
    if args.model == 'resnet50':
        return models.resnet50()
