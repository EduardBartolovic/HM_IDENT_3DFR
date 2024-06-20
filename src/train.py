import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from config import configurations
from backbone.model_resnet import ResNet_50, ResNet_101, ResNet_152
from backbone.model_irse import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
from head.metrics import ArcFace, CosFace, SphereFace, Am_softmax
from loss.focal import FocalLoss
from src.backbone.model_irse_rgbd import IR_152_rgbd, IR_101_rgbd, IR_50_rgbd, IR_SE_50_rgbd, IR_SE_101_rgbd, \
    IR_SE_152_rgbd
from src.backbone.model_resnet_rgbd import ResNet_50_rgbd, ResNet_101_rgbd, ResNet_152_rgbd
from src.util.ImageFolder4Channel import ImageFolder4Channel
from src.util.misc import colorstr
from util.eval_model import evaluate_and_log
from util.utils import make_weights_for_balanced_classes, separate_irse_bn_paras, \
    separate_resnet_bn_paras, warm_up_lr, schedule_lr, AverageMeter, accuracy

from tensorboardX import SummaryWriter
from tqdm import tqdm
import os
import mlflow

if __name__ == '__main__':

    # ======= hyperparameters & data loaders =======#
    cfg = configurations[1]

    SEED = cfg['SEED']  # random seed for reproduce results
    torch.manual_seed(SEED)

    RUN_NAME = cfg['RUN_NAME']
    DATA_ROOT = cfg['DATA_ROOT']  # the parent root where your train/val/test data are stored
    TRAIN_SET = cfg['TRAIN_SET']
    MODEL_ROOT = cfg['MODEL_ROOT']  # the root to buffer your checkpoints
    LOG_ROOT = cfg['LOG_ROOT']  # the root to log your train/val status
    BACKBONE_RESUME_ROOT = cfg['BACKBONE_RESUME_ROOT']  # the root to resume training from a saved checkpoint
    HEAD_RESUME_ROOT = cfg['HEAD_RESUME_ROOT']  # the root to resume training from a saved checkpoint

    BACKBONE_NAME = cfg['BACKBONE_NAME']  # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
    HEAD_NAME = cfg['HEAD_NAME']  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
    LOSS_NAME = cfg['LOSS_NAME']  # support: ['Focal', 'Softmax']
    DISTANCE_METRIC = cfg['DISTANCE_METRIC']  # support: ['euclidian', 'cosine']

    INPUT_SIZE = cfg['INPUT_SIZE']
    RGB_MEAN = cfg['RGB_MEAN']  # for normalize inputs
    RGB_STD = cfg['RGB_STD']
    EMBEDDING_SIZE = cfg['EMBEDDING_SIZE']  # feature dimension
    BATCH_SIZE = cfg['BATCH_SIZE']
    DROP_LAST = cfg['DROP_LAST']  # whether drop the last batch to ensure consistent batch_norm statistics
    LR = cfg['LR']  # initial LR
    NUM_EPOCH = cfg['NUM_EPOCH']
    WEIGHT_DECAY = cfg['WEIGHT_DECAY']
    MOMENTUM = cfg['MOMENTUM']
    STAGES = cfg['STAGES']  # epoch stages to decay learning rate

    DEVICE = cfg['DEVICE']
    MULTI_GPU = cfg['MULTI_GPU']  # flag to use multiple GPUs
    GPU_ID = cfg['GPU_ID']  # specify your GPU ids
    PIN_MEMORY = cfg['PIN_MEMORY']
    NUM_WORKERS = cfg['NUM_WORKERS']
    print("=" * 60)
    print("Overall Configurations:")
    print(cfg)
    print("=" * 60)

    mlflow.set_tracking_uri(f'file:{LOG_ROOT}/mlruns')
    mlflow.set_experiment(RUN_NAME)
    with mlflow.start_run(run_name=RUN_NAME) as run:

        mlflow.log_param('config', cfg)
        print(run.info.run_id)
        log_dir = f'{LOG_ROOT}/tensorboard/{RUN_NAME}'
        writer = SummaryWriter(log_dir)

        # refer to https://pytorch.org/docs/stable/torchvision/transforms.html for more online data augmentation
        train_transform = transforms.Compose([
            transforms.Resize([int(128 * INPUT_SIZE[0] / 112), int(128 * INPUT_SIZE[0] / 112)]),  # smaller side resized
            transforms.RandomCrop([INPUT_SIZE[0], INPUT_SIZE[1]]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=RGB_MEAN, std=RGB_STD),
        ])

        if 'rgbd' in TRAIN_SET:
            dataset_train = ImageFolder4Channel(os.path.join(DATA_ROOT, TRAIN_SET), train_transform)
        else:
            dataset_train = datasets.ImageFolder(os.path.join(DATA_ROOT, TRAIN_SET), train_transform)

        # create a weighted random sampler to process imbalanced data
        weights = make_weights_for_balanced_classes(dataset_train.imgs, len(dataset_train.classes))
        weights = torch.DoubleTensor(weights)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

        train_loader = torch.utils.data.DataLoader(
            dataset_train, batch_size=BATCH_SIZE, sampler=sampler, pin_memory=PIN_MEMORY,
            num_workers=NUM_WORKERS, drop_last=DROP_LAST
        )

        NUM_CLASS = len(train_loader.dataset.classes)
        print("Number of Training Classes: {}".format(NUM_CLASS))

        # ======= model & loss & optimizer =======#
        BACKBONE_DICT = {'ResNet_50': ResNet_50(INPUT_SIZE),
                         'ResNet_101': ResNet_101(INPUT_SIZE),
                         'ResNet_152': ResNet_152(INPUT_SIZE),
                         'ResNet_50_RGBD': ResNet_50_rgbd(INPUT_SIZE),
                         'ResNet_101_RGBD': ResNet_101_rgbd(INPUT_SIZE),
                         'ResNet_152_RGBD': ResNet_152_rgbd(INPUT_SIZE),
                         'IR_50': IR_50(INPUT_SIZE),
                         'IR_101': IR_101(INPUT_SIZE),
                         'IR_152': IR_152(INPUT_SIZE),
                         'IR_50_RGBD': IR_50_rgbd(INPUT_SIZE),
                         'IR_101_RGBD': IR_101_rgbd(INPUT_SIZE),
                         'IR_152_RGBD': IR_152_rgbd(INPUT_SIZE),
                         'IR_SE_50': IR_SE_50(INPUT_SIZE),
                         'IR_SE_101': IR_SE_101(INPUT_SIZE),
                         'IR_SE_152': IR_SE_152(INPUT_SIZE),
                         'IR_SE_50_RGBD': IR_SE_50_rgbd(INPUT_SIZE),
                         'IR_SE_101_RGBD': IR_SE_101_rgbd(INPUT_SIZE),
                         'IR_SE_152_RGBD': IR_SE_152_rgbd(INPUT_SIZE)}
        if 'rgbd' in TRAIN_SET:
            BACKBONE_NAME = BACKBONE_NAME + '_RGBD'
        BACKBONE = BACKBONE_DICT[BACKBONE_NAME]
        print("=" * 60)
        print(colorstr('magenta', BACKBONE))
        print(colorstr('blue', f"{BACKBONE_NAME} Backbone Generated"))
        print("=" * 60)

        HEAD_DICT = {'ArcFace': ArcFace(in_features=EMBEDDING_SIZE, out_features=NUM_CLASS, device_id=GPU_ID),
                     'CosFace': CosFace(in_features=EMBEDDING_SIZE, out_features=NUM_CLASS, device_id=GPU_ID),
                     'SphereFace': SphereFace(in_features=EMBEDDING_SIZE, out_features=NUM_CLASS, device_id=GPU_ID),
                     'Am_softmax': Am_softmax(in_features=EMBEDDING_SIZE, out_features=NUM_CLASS, device_id=GPU_ID)}
        HEAD = HEAD_DICT[HEAD_NAME]
        print(colorstr('magenta', HEAD))
        print(colorstr('blue', f"{HEAD_NAME} Head Generated"))
        print("=" * 60)

        LOSS_DICT = {'Focal': FocalLoss(),
                     'Softmax': nn.CrossEntropyLoss(),
                     # 'AdaCos' : AdaCos(),
                     # 'AdaM_Softmax': AdaM_Softmax() ,
                     'ArcFace': ArcFace,
                     # 'ArcNegFace': ArcNegFace(),
                     # 'CircleLoss': Circleloss(),
                     # 'CurricularFace': CurricularFace(),
                     # 'MagFace' :  MagFace(),
                     # 'NPCFace' :  MV_Softmax.py(),
                     # 'SST_Prototype': SST_Prototype()
                     }
        LOSS = LOSS_DICT[LOSS_NAME]
        print(colorstr('magenta', LOSS))
        print(colorstr('blue', f"{LOSS_NAME} Loss Generated"))
        print("=" * 60)

        if BACKBONE_NAME.find("IR") >= 0:
            backbone_paras_only_bn, backbone_paras_wo_bn = separate_irse_bn_paras(BACKBONE)  # separate batch_norm parameters from others; do not do weight decay for batch_norm parameters to improve the generalizability
            _, head_paras_wo_bn = separate_irse_bn_paras(HEAD)
        else:
            backbone_paras_only_bn, backbone_paras_wo_bn = separate_resnet_bn_paras(BACKBONE)  # separate batch_norm parameters from others; do not do weight decay for batch_norm parameters to improve the generalizability
            _, head_paras_wo_bn = separate_resnet_bn_paras(HEAD)
        OPTIMIZER = optim.SGD([{'params': backbone_paras_wo_bn + head_paras_wo_bn, 'weight_decay': WEIGHT_DECAY},
                               {'params': backbone_paras_only_bn}], lr=LR, momentum=MOMENTUM)
        print(colorstr('magenta', OPTIMIZER))
        print(colorstr('blue', "Optimizer Generated"))
        print("=" * 60)

        # optionally resume from a checkpoint
        if BACKBONE_RESUME_ROOT and HEAD_RESUME_ROOT:
            print("=" * 60)
            if os.path.isfile(BACKBONE_RESUME_ROOT) and os.path.isfile(HEAD_RESUME_ROOT):
                print(colorstr('blue', f"Loading Backbone Checkpoint {BACKBONE_RESUME_ROOT}"))
                BACKBONE.load_state_dict(torch.load(BACKBONE_RESUME_ROOT))
                print(colorstr('blue', f"Loading Head Checkpoint {HEAD_RESUME_ROOT}"))
                HEAD.load_state_dict(torch.load(HEAD_RESUME_ROOT))
            if os.path.isfile(BACKBONE_RESUME_ROOT):
                print(colorstr('blue', f"Loading ONLY Backbone Checkpoint {BACKBONE_RESUME_ROOT}"))
                BACKBONE.load_state_dict(torch.load(BACKBONE_RESUME_ROOT))
            else:
                print(colorstr('red', f"No Checkpoint Found at {BACKBONE_RESUME_ROOT} and {HEAD_RESUME_ROOT}. Please Have a Check or Continue to Train from Scratch"))
            print("=" * 60)

        if MULTI_GPU:
            # multi-GPU setting
            BACKBONE = nn.DataParallel(BACKBONE, device_ids=GPU_ID)
            BACKBONE = BACKBONE.to(DEVICE)
        else:
            # single-GPU setting
            BACKBONE = BACKBONE.to(DEVICE)

        # ======= train & validation & save checkpoint =======#
        DISP_FREQ = len(train_loader) // 5  # frequency to display training loss & acc # was 100

        NUM_EPOCH_WARM_UP = NUM_EPOCH // 25  # use the first 1/25 epochs to warm up
        NUM_BATCH_WARM_UP = len(train_loader) * NUM_EPOCH_WARM_UP  # use the first 1/25 epochs to warm up
        batch = 0  # batch index

        # ======= Initialize early stopping parameters =======#
        patience = 10  # Number of epochs to wait after last time validation loss improved.
        best_acc = 0  # Initial best value
        counter = 0  # Counter for epochs without improvement

        for epoch in range(NUM_EPOCH):  # start training process
            # adjust LR for each training stage after warm up, you can also choose to adjust LR manually (with slight modification) once plateau observed
            if epoch == STAGES[0]:
                schedule_lr(OPTIMIZER)
            if epoch == STAGES[1]:
                schedule_lr(OPTIMIZER)
            if epoch == STAGES[2]:
                schedule_lr(OPTIMIZER)

            BACKBONE.train()  # set to training mode
            HEAD.train()

            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()

            for inputs, labels in tqdm(iter(train_loader)):

                if (epoch + 1 <= NUM_EPOCH_WARM_UP) and (
                        batch + 1 <= NUM_BATCH_WARM_UP):  # adjust LR for each training batch during warm up
                    warm_up_lr(batch + 1, NUM_BATCH_WARM_UP, LR, OPTIMIZER)

                # compute output
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE).long()
                features = BACKBONE(inputs)
                outputs = HEAD(features, labels)
                loss = LOSS(outputs, labels)

                # measure accuracy and record loss
                prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))
                losses.update(loss.data.item(), inputs.size(0))
                top1.update(prec1.data.item(), inputs.size(0))
                top5.update(prec5.data.item(), inputs.size(0))

                # compute gradient and do SGD step
                OPTIMIZER.zero_grad()
                loss.backward()
                OPTIMIZER.step()

                # display training loss & acc every DISP_FREQ
                if ((batch + 1) % DISP_FREQ == 0) and batch != 0:
                    print("=" * 60)
                    print(colorstr('cyan',
                                   f'Epoch {epoch + 1}/{NUM_EPOCH} Batch {batch + 1}/{len(train_loader) * NUM_EPOCH}\t'
                                   f'Training Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                                   f'Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                                   f'Training Prec@5 {top5.val:.3f} ({top5.avg:.3f})'))
                    print("=" * 60)

                batch += 1  # batch index

            # training statistics per epoch (buffer for visualization)
            epoch_loss = losses.avg
            epoch_acc = top1.avg
            writer.add_scalar("Training_Loss", epoch_loss, epoch + 1)
            writer.add_scalar("Training_Accuracy", epoch_acc, epoch + 1)
            mlflow.log_metric('train_loss', epoch_loss, step=epoch + 1)
            mlflow.log_metric('Training_Accuracy', epoch_acc, step=epoch + 1)
            print("#" * 60)
            print(colorstr('bright_green', f'Epoch: {epoch + 1}/{NUM_EPOCH}\t'
                                           f'Training Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                                           f'Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                                           f'Training Prec@5 {top5.val:.3f} ({top5.avg:.3f})'))
            print("#" * 60)

            #  ======= perform validation =======
            if 'rgbd' in TRAIN_SET:
                test_bellus = 'test_rgbd_bellus'
                test_facescape = 'test_rgbd_facescape'
                test_faceverse = 'test_rgbd_faceverse'
                test_texas = 'test_rgbd_texas'
            elif 'rgb' in TRAIN_SET:
                test_bellus = 'test_rgb_bellus'
                test_facescape = 'test_rgb_facescape'
                test_faceverse = 'test_rgb_faceverse'
                test_texas = 'test_rgb_texas'
            elif 'depth' in TRAIN_SET:
                test_bellus = 'test_depth_bellus'
                test_facescape = 'test_depth_facescape'
                test_faceverse = 'test_depth_faceverse'
                test_texas = 'test_depth_texas'

            evaluate_and_log(DEVICE, BACKBONE, DATA_ROOT, test_bellus, writer, epoch, NUM_EPOCH, DISTANCE_METRIC, RGB_MEAN, RGB_STD)
            if (epoch + 1) % 5 == 0:
                evaluate_and_log(DEVICE, BACKBONE, DATA_ROOT, test_facescape, writer, epoch, NUM_EPOCH, DISTANCE_METRIC, RGB_MEAN, RGB_STD)
                evaluate_and_log(DEVICE, BACKBONE, DATA_ROOT, test_faceverse, writer, epoch, NUM_EPOCH, DISTANCE_METRIC, RGB_MEAN, RGB_STD)
                evaluate_and_log(DEVICE, BACKBONE, DATA_ROOT, test_texas, writer, epoch, NUM_EPOCH, DISTANCE_METRIC, RGB_MEAN, RGB_STD)

            print("=" * 60)

            # Early stopping check
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(colorstr('red', "=" * 60))
                    print(colorstr('red', " ======== Early stopping triggered ======== "))
                    print(colorstr('red', "=" * 60))
                    break

            time.sleep(0.3)
            # save checkpoints per epoch
            # if MULTI_GPU:
            #     torch.save(BACKBONE.module.state_dict(), os.path.join(MODEL_ROOT, "Backbone_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(BACKBONE_NAME, epoch + 1, batch, get_time())))
            #     torch.save(HEAD.state_dict(), os.path.join(MODEL_ROOT, "Head_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(HEAD_NAME, epoch + 1, batch, get_time())))
            # else:
            #     torch.save(BACKBONE.state_dict(), os.path.join(MODEL_ROOT, "Backbone_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(BACKBONE_NAME, epoch + 1, batch, get_time())))
            #     torch.save(HEAD.state_dict(), os.path.join(MODEL_ROOT, "Head_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(HEAD_NAME, epoch + 1, batch, get_time())))
