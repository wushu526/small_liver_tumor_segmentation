import logging
import os
import sys
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from utils.eval import eval_net
from unet import UNet, UNet_2, UNet_3, UNet_4
from utils.loss import FocalTversky_BCELoss, FocalTverskyLoss
from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split

from torchsummary import summary
# choose the gpu id
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device = torch.device("cuda")
# the training set & the training label

# model saving path
file_name = os.path.basename(__file__).split('.')[0]
dir_checkpoint = f"checkpoints/{file_name}/"


def multi_loss_compute(multi_sv, masks_pred, true_masks, criterion, writer, global_step, pbar, optimizer):
    postfix = {}
    loss = []
    for i in range(multi_sv):
        loss.append(criterion(masks_pred[i], true_masks))
        num = str(i + 1)
        writer.add_scalar("Train/loss" + num, loss[i].item(), global_step)
        postfix["loss" + num + " (batch)"] = loss[i].item()
    #清空之前存储的梯度
    optimizer.zero_grad()
    for i in range(multi_sv):
        if ((i + 1) != multi_sv):
            loss[i].backward(retain_graph=True)
        else:
            loss[i].backward()
    pbar.set_postfix(postfix)



def train_net(net, device, epochs, batch_size, lr, val_percent, save_cp, n_classes, multi_sv):
    dataset = BasicDataset(imgs_dir=dir_img, masks_dir=dir_mask, n_classes=n_classes, transform=True)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=32, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=32, pin_memory=True, drop_last=True)

    # 训练时改动log名
    writer = SummaryWriter(comment=f"_{file_name}")
    global_step = 0
    best_val_score = 0

    logging.info(f"""Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
    """)

    optimizer = optim.Adam(net.parameters(), lr=lr)
    # optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, 'min' if net.n_classes > 1 else 'max', patience=5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)  # 固定步长衰减
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = FocalTversky_BCELoss().cuda()
        # criterion = FocalTverskyLoss().cuda()
        # criterion = nn.BCEWithLogitsLoss().cuda()

    for epoch in range(epochs):
        net.train()
        with tqdm(total=n_train, desc=f"Epoch {epoch + 1}/{epochs}", unit="img") as pbar:
            for batch in train_loader:
                imgs = batch["image"]
                true_masks = batch["mask"]

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                # 向前传播
                masks_pred = net(imgs)
                # 计算损失
                multi_loss_compute(multi_sv, masks_pred, true_masks, criterion, writer, global_step, pbar, optimizer)
                # 梯度裁剪
                nn.utils.clip_grad_value_(net.parameters(), 0.01)
                # 将梯度应用于参数更新
                optimizer.step()
                pbar.update(imgs.shape[0])
                global_step += 1

                if global_step % (n_train // (2 * batch_size)) == 0:
                    val_score = eval_net(net, val_loader, device)
                    writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch + 1)
                    if net.n_classes > 1:
                        logging.info('Validation cross entropy: {}'.format(val_score))
                        writer.add_scalar('Loss/val', val_score, global_step)
                    else:
                        logging.info('Validation Dice Coeff: {}'.format(val_score))
                        writer.add_scalar('Dice/val', val_score, global_step)
        val_score = eval_net(net, val_loader, device)
        scheduler.step()
        if val_score > best_val_score:
            best_val_score = val_score
            if save_cp:
                try:
                    os.mkdir(dir_checkpoint)
                    logging.info("Created checkpoint directory")
                except OSError:
                    pass
                torch.save(net.state_dict(), dir_checkpoint + f"CP_epoch{epoch+1}_dice{best_val_score}.pth")
                logging.info(f"Checkpoint {epoch+1}_dice{best_val_score} saved !")

    writer.close()


def weight_init(m):
    for m in m.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)


def get_args(input_path, label_path, model_type, epochs, batchsize, lr, val, channels, classes, load=None):
    dict = {}
    dict['input_path'] = input_path
    dict['label_path'] = label_path
    dict['model_type'] = model_type
    dict['epochs'] = epochs
    dict['batchsize'] = batchsize
    dict['lr'] = lr
    dict['val'] = val
    dict['channels'] = channels
    dict['classes'] = classes
    dict['load'] = load
    return dict


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.info(f"Using device {device}")
    args = get_args(input_path=
                    '/home3/HWGroup/wushu/liver_tumor/data/v1.4.0/preprocessed/imagesTr_2D_npy/',
                    label_path=
                    '/home3/HWGroup/wushu/liver_tumor/data/v1.4.0/preprocessed/labelsTr_2D_npy/',
                    load=
                    '',
                    # model_type=UNet_2,
                    model_type=UNet_4,
                    epochs=500,
                    batchsize=32,
                    lr=0.001,
                    val=10.0,
                    channels=1,
                    classes=1)
    dir_img = args['input_path']
    dir_mask = args['label_path']
    # choose the model type
    net = args['model_type'](n_channels=args['channels'], n_classes=args['classes'], bilinear=False)
    multi_sv = int(str(args['model_type']).split('_')[-1][:-2])
    # the parameters initialization
    net.apply(weight_init)
    net.to(device=device)

    # record the info
    logging.info(f"Network:\n"
                 f"\t{args['channels']} input channels\n"
                 f"\t{args['classes']} output channels (classes)")

    # load the model to finetune
    if args['load']:
        net.load_state_dict(torch.load(args['load'], map_location=device))
        logging.info(f"Model loaded from {args['load']}")

    print(summary(net,(1,400,400),32))

    # start to train the model
    try:
        train_net(net=net,
                  epochs=args['epochs'],
                  batch_size=args['batchsize'],
                  lr=args['lr'],
                  device=device,
                  save_cp=True,
                  val_percent=args['val'] / 100,
                  n_classes=args['classes'],
                  multi_sv=multi_sv)
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
