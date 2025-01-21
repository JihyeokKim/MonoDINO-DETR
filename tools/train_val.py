import warnings
warnings.filterwarnings("ignore")

import os
import sys
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import yaml
import argparse
import datetime

from lib.helpers.model_helper import build_model
from lib.helpers.dataloader_helper import build_dataloader, build_testloader, make_data_loader, make_test_loader
from lib.helpers.optimizer_helper import build_optimizer
from lib.helpers.scheduler_helper import build_lr_scheduler
from lib.helpers.trainer_helper import Trainer
from lib.helpers.tester_helper import Tester
from lib.helpers.utils_helper import create_logger, log_config_to_file
from lib.helpers.utils_helper import set_random_seed
from lib.helpers.config import cfg, cfg_from_yaml_file
from lib.helpers import utils_helper
from tensorboardX import SummaryWriter

import torch.distributed as dist

from lib.helpers import launch, comm

def parse_config():
    parser = argparse.ArgumentParser(description='Depth-aware Transformer for Monocular 3D Object Detection')
    parser.add_argument('--config', dest='config', help='settings of detection in yaml format')
    parser.add_argument('-e', '--evaluate_only', action='store_true', default=False, help='evaluation only')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')

    parser.add_argument('--model_name', type=str, default='default', help='extra tag for this experiment')
    # parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--num_gpus', type=int, default=1, help='number of gpus')
    parser.add_argument('--num_machines', type=int, default=1, help='number of machines')
    parser.add_argument('--dist_url', type=str, default='auto', help='url used to set up distributed training')
    parser.add_argument('--max_epoch', type=int, default=195, help='max epochs for training')

    args = parser.parse_args()

    return args
    

def main(args):
    cfg_from_yaml_file(args.config, cfg)
    
    distributed = comm.get_world_size() > 1
    if not distributed:
        args.sync_bn = False
        total_gpus = 1
    else:
        total_gpus = comm.get_world_size()
        cfg.local_rank = comm.get_local_rank()
        

    if args.batch_size is None:
        args.batch_size = cfg.trainer.batch_size_per_gpu
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    cfg.trainer.max_epoch = args.max_epoch

    set_random_seed(cfg.get('random_seed', 444) + cfg.local_rank)

    model_name = args.model_name
    output_path = os.path.join('./' + cfg["trainer"]['save_path'], model_name)
    os.makedirs(output_path, exist_ok=True)

    log_file = os.path.join(output_path, 'train.log.%s' % datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    logger = create_logger(log_file, rank=cfg.local_rank)

    # build model
    model, loss = build_model(cfg['model'])

    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    device = torch.device("cuda", cfg.local_rank)
    # gpu_ids = list(map(int, cfg['trainer']['gpu_ids'].split(',')))
    gpu_ids = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_ids)
    
    # Initialize the model
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.local_rank)
    torch.cuda.set_device(cfg.local_rank)
    torch.cuda.empty_cache()
    model.to(cfg.local_rank)

    if args.evaluate_only:
        test_set, test_loader, test_sampler = build_testloader(cfg['dataset'],
                                              batch_size=args.batch_size,
                                              dist=distributed,
                                              training=False)
        
        logger.info('###################  Evaluation Only  ##################')
        tester = Tester(cfg=cfg['tester'],
                        model=model,
                        dataloader=test_loader,
                        logger=logger,
                        train_cfg=cfg['trainer'],
                        model_name=model_name,
                        dist_test=distributed,
                        rank=cfg.local_rank,
                        batch_size=args.batch_size)
        tester.test()
        return
    
    if distributed:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
        
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)
    if cfg.local_rank == 0:
        os.system('cp %s %s' % (args.config, output_path))

    tb_log = SummaryWriter(log_dir=str(output_path + '/tensorboard')) if cfg.local_rank == 0 else None

    model.train()
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.local_rank % torch.cuda.device_count()], find_unused_parameters=True)

    #  build optimizer
    optimizer = build_optimizer(cfg['optimizer'], model)

    # build dataloader
    train_set, train_loader, train_sampler = build_dataloader(cfg['dataset'],
                                                 batch_size=args.batch_size,
                                                 dist=distributed,
                                                 training=True)
    
    test_set, test_loader, test_sampler = build_testloader(cfg['dataset'],
                                              batch_size=args.batch_size,
                                              dist=distributed,
                                              training=False)
    
    
    # build lr scheduler
    lr_scheduler, warmup_lr_scheduler = build_lr_scheduler(cfg['lr_scheduler'], optimizer, last_epoch=-1)
    trainer = Trainer(cfg=cfg['trainer'],
                      model=model,
                      optimizer=optimizer,
                      train_loader=train_loader,
                      lr_scheduler=lr_scheduler,
                      warmup_lr_scheduler=warmup_lr_scheduler,
                      logger=logger,
                      loss=loss,
                      model_name=model_name,
                      dist_train=distributed,
                      tb_log=tb_log,
                      rank=cfg.local_rank)

    tester = Tester(cfg=cfg['tester'],
                    model=model,
                    dataloader=test_loader,
                    logger=logger,
                    train_cfg=cfg['trainer'],
                    model_name=model_name,
                    dist_test=distributed,
                    rank=cfg.local_rank,
                    batch_size=args.batch_size)
    
    if cfg['dataset']['test_split'] != 'test':
        trainer.tester = tester

    logger.info('###################  Training  ##################')
    logger.info('Batch Size: %d' % (args.batch_size))
    logger.info('Learning Rate: %f' % (cfg['optimizer']['lr']))

    trainer.train()

    if cfg['dataset']['test_split'] == 'test':
        return

    if distributed:
        dist.destroy_process_group()
        logger.info('Destroyed the distributed process group.')
        distributed = comm.get_world_size() > 1
        logger.info('Distributed: %s' % (distributed))

    

    if cfg.local_rank == 0:
        logger.info('Rebuilding the model without DistributedDataParallel for testing.')
        
        # Rebuild the model
        model, loss = build_model(cfg['model'])
        model.to(device)
        model.eval()
        
        # Optionally, disable gradient computation for testing
        torch.set_grad_enabled(False)
        
        # Rebuild the test loader
        test_set, test_loader, test_sampler = build_testloader(cfg['dataset'],
                                              batch_size=args.batch_size,
                                              dist=distributed,
                                              training=False)
        
        # Rebuild the tester without DistributedDataParallel
        tester = Tester(cfg=cfg['tester'],
                        model=model,
                        dataloader=test_loader,
                        logger=logger,
                        train_cfg=cfg['trainer'],
                        model_name=model_name,
                        dist_test=False,
                        rank=cfg.local_rank,
                        batch_size=args.batch_size)
        
        logger.info('###################  Testing  ##################')
        logger.info('Batch Size: %d' % (args.batch_size))
        logger.info('Split: %s' % (cfg['dataset']['test_split']))
        
        # Perform testing
        tester.test()
        
        # # Re-enable gradient computation if necessary
        # torch.set_grad_enabled(True)
        
        logger.info('Testing completed successfully.')
        logger.info('Exiting the program.')

    return


        

if __name__ == '__main__':
    args = parse_config()

    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.local_rank,
        dist_url=args.dist_url,
        args=(args,),        
    )
