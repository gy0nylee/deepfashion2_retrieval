import torch.optim as optim

def get_optimizer(params, args):
    if args.optim == 'adam':
        return optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay, eps=args.eps)
    if args.optim == 'sgd':
        return optim.SGD(params, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    if args.optim == 'rmsprop':
        return optim.RMS(params, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum, eps=args.eps)
    if args.optim == 'adagrad':
        return optim.Adagrad(params, lr=args.lr, weight_decay=args.weight_decay, eps=args.eps)
