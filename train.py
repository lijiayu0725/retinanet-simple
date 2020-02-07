import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

from data import DataIterator
from model import RetinaNet


def train():
    model = RetinaNet(state_dict_path='/home/lijiayu/.cache/torch/checkpoints/resnet50-19c8e357.pth')
    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = SGD(model.parameters(), lr=0.0025, weight_decay=1e-4, momentum=0.9)

    model.train()

    warmup = 1000
    gamma = 0.1
    milestores = [60000, 80000]
    iterations = 90000
    batch_size = 2

    def schedule(iter):
        if iter <= warmup:
            return 0.9 * iter / warmup + 0.1
        return gamma ** len([m for m in milestores if m <= iter])

    scheduler = LambdaLR(optimizer, schedule)

    print('preparing dataset...')
    data_iterator = DataIterator(path='/data/datasets/coco2017', batch_size=batch_size, resize=600)
    print('finish loading dataset!')
    print('starting training...')
    iteration = 0

    while iteration < iterations:
        cls_losses, box_losses = [], []
        for data, target in data_iterator:
            iteration += 1

            optimizer.zero_grad()
            cls_loss, box_loss = model([data, target])
            loss = cls_loss + box_loss

            loss.backward()
            optimizer.step()
            scheduler.step(iteration)

            cls_losses.append(cls_loss.mean().item())
            box_losses.append(box_loss.mean().item())

            if not torch.isfinite(loss).all():
                raise RuntimeError('Loss is diverging!')

            if iteration % 10 == 0:
                focal_loss = torch.FloatTensor(cls_losses).mean().item()
                box_loss = torch.FloatTensor(box_losses).mean().item()
                learning_rate = optimizer.param_groups[0]['lr']

                msg = '[{:{len}}/{}]'.format(iteration, iterations, len=len(str(iterations)))
                msg += ' focal loss: {:.3f}'.format(focal_loss)
                msg += ', box loss: {:.3f}'.format(box_loss)
                msg += ', lr: {:.2g}'.format(learning_rate)
                print(msg)

                cls_losses, box_losses = [], []

                if iteration % 1000 == 0:
                    print('saving model for iter {}'.format(iteration))
                    torch.save(model.state_dict(), './checkpoints/iter-{}.pth'.format(iteration))

            if iteration == iterations:
                break

    print('finish training, saving the final model...')
    torch.save(model.state_dict(), './checkpoints/final.pth')
    print('-' * 10 + 'completed!' + '-' * 10)


if __name__ == '__main__':
    train()
