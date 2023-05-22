import os
import random
import thop
import torch

from models import model_mnet
from utils import logger, line_seg

__all__ = ["init_device", "init_model"]


def init_device(seed=None, cpu=None, gpu=None, affinity=None):
    # set the CPU affinity
    if affinity is not None:
        os.system(f'taskset -p {affinity} {os.getpid()}')

    # Set the random seed
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    # Set the GPU id you choose
    if gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    # Env setup
    if not cpu and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
        if seed is not None:
            torch.cuda.manual_seed(seed)
        pin_memory = True
        logger.info("Running on GPU%d" % (gpu if gpu else 0))
    else:
        pin_memory = False
        device = torch.device('cpu')
        logger.info("Running on CPU")

    return device, pin_memory


def load_pretrained_model(model, pretrained_dict):
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)


def init_model(args):
    # Model loading
    model = model_mnet(reduction=args.cr, expansion=args.expansion)

    if args.pretrained is not None:
        assert os.path.isfile(args.pretrained)
        checkpoints = torch.load(args.pretrained)
        load_pretrained_model(model, checkpoints['state_dict'])
        logger.info("pretrained model loaded from {}".format(args.pretrained))

    # Model flops and params counting
    image = torch.randn([1, 2, 32, 32])
    flops, params = thop.profile(model, inputs=(image,), verbose=False)
    flops, params = thop.clever_format([flops, params], "%.3f")

    # Model info logging
    logger.info(f'=> Model Name: M-Net [pretrained: {args.pretrained}]')
    logger.info(f'=> Model Config: compression ratio=1/{args.cr}')
    logger.info(f'=> Model Flops: {flops}')
    logger.info(f'=> Model Params Num: {params}\n')
    logger.info(f'{line_seg}\n{model}\n{line_seg}\n')

    return model
