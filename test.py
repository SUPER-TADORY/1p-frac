import hydra
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from timm.utils import accuracy

from data import create_dataloader
from logger import MetricLogger, SmoothedValue
from utils import init_distributed_mode


torch.backends.cudnn.benchmark = True


@torch.no_grad()
def evaluate(dataloader, model, n_iter):
    criterion = torch.nn.CrossEntropyLoss().cuda()
    model.eval()

    evalloagger = MetricLogger(delimiter=' ')
    evalloagger.add_meter('eval_loss', SmoothedValue(window_size=1, fmt='{global_avg:.4f}'))
    evalloagger.add_meter('eval_acc1', SmoothedValue(window_size=1, fmt='{value:.3f}'))
    
    header = 'Val:'
    for data in evalloagger.log_every(dataloader, n_iter, 10, header):
        images = data[0].cuda(non_blocking=True)
        labels = data[1].cuda(non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs = model(images)

        eval_loss = criterion(outputs, labels)
        acc1, _ = accuracy(outputs, labels, topk=(1, 5))

        torch.cuda.synchronize()
        batch_size = images.size(0)
        evalloagger.update(eval_loss=eval_loss.item())
        evalloagger.update(eval_acc1=acc1.item(), n=batch_size)

    evalloagger.synchronize_between_processes()
    print('* Acc@1: {top1.global_avg:.3f} Eval loss: {losses.global_avg:.3f}'.format(top1=evalloagger.eval_acc1, losses=evalloagger.eval_loss))


@hydra.main(config_path='./configs', config_name='test')
def main(cfg):
    # Initialize torch.distributed using MPI
    init_distributed_mode(cfg.dist)

    # Create Dataloader
    world_size = dist.get_world_size()
    total_batch_size = cfg.data.loader.batch_size * world_size
    valloader = create_dataloader(cfg.data, is_training=False)
    n_val_iter = cfg.data.baseinfo.val_imgs // total_batch_size

    # model 
    model = instantiate(cfg.model.arch, num_classes=cfg.data.baseinfo.num_classes)
    print(f'Model[{cfg.model.arch.model_name}] was created')

    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.dist.local_rank])
    model_without_ddp = model.module

    # Load trained weights
    checkpoint = torch.load(cfg.ckpt, map_location='cpu')
    model_without_ddp.load_state_dict(checkpoint['model'])
    print(f'Checkpoint was loaded from {cfg.ckpt}\n')

    evaluate(valloader, model, n_val_iter)



if __name__ == '__main__':
    main()
