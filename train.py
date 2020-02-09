import torch
from torch.nn import DataParallel
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

from data import DataIterator
from model import RetinaNet


def train():
    warmup = 1000
    gamma = 0.1
    milestores = [8, 11]
    batch_size = 16
    stride = 32
    lr = 0.01
    weight_decay = 1e-4
    momentem = 0.9
    epochs = 12
    shuffle = True
    resize = 600
    resnet_dir = '/home/lijiayu/.cache/torch/checkpoints/resnet50-19c8e357.pth'
    coco_dir = '/data/datasets/coco2017'
    mb_to_gb_factor = 1024 ** 3

    model = RetinaNet(state_dict_path=resnet_dir)
    if torch.cuda.is_available():
        model = model.cuda()
    model = DataParallel(model)
    optimizer = SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentem)

    print('preparing dataset...')
    data_iterator = DataIterator(path=coco_dir, batch_size=batch_size, stride=stride, shuffle=shuffle, resize=resize)
    print('finish loading dataset!')

    model.train()

    def schedule_warmup(i):
        return 0.9 * i / warmup + 0.1

    def schedule(epoch):
        return gamma ** len([m for m in milestores if m <= epoch])

    scheduler_warmup = LambdaLR(optimizer, schedule_warmup)
    scheduler = LambdaLR(optimizer, schedule)

    print('starting training...')

    for epoch in range(1, epochs + 1):
        cls_losses, box_losses = [], []
        for i, (data, target) in enumerate(data_iterator, start=1):
            optimizer.zero_grad()
            cls_loss, box_loss = model((data, target))

            loss = cls_loss + box_loss
            loss.sum().backward()
            optimizer.step()
            if epoch == 1 and i <= warmup:
                scheduler_warmup.step(i)

            cls_losses.append(cls_loss.mean().item())
            box_losses.append(box_loss.mean().item())

            if not torch.isfinite(loss).all():
                raise RuntimeError('Loss is diverging!')

            if i % 10 == 0:
                focal_loss = torch.FloatTensor(cls_losses).mean().item()
                box_loss = torch.FloatTensor(box_losses).mean().item()
                learning_rate = optimizer.param_groups[0]['lr']

                msg = '[{:{len}}/{}]'.format(epoch, epochs, len=len(str(epochs)))
                msg += '[{:{len}}/{}]'.format(i, len(data_iterator), len=len(str(len(data_iterator))))
                msg += ' focal loss: {:.3f}'.format(focal_loss)
                msg += ', box loss: {:.3f}'.format(box_loss)
                msg += ', lr: {:.2g}'.format(learning_rate)
                msg += ', cuda_memory: {:.3g} GB'.format(torch.cuda.max_memory_allocated() / mb_to_gb_factor)
                print(msg)
                del loss, cls_losses[:], box_losses[:]
                cls_losses, box_losses = [], []
            del cls_loss, box_loss, data, target

        scheduler.step(epoch)
        print('saving model for epoch {}'.format(epoch))
        torch.save(model.state_dict(), './checkpoints/epoch-{}.pth'.format(epoch))


    print('finish training, saving the final model...')
    torch.save(model.state_dict(), './checkpoints/final.pth')
    print('-' * 10 + 'completed!' + '-' * 10)


if __name__ == '__main__':
    train()
