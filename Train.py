import os
import time
import torch
import argparse
import torchvision
import torch.nn as nn
import torch.optim as optimize
import torch.utils.data as data

from CSNet import CSNet

""" the local ../images/test/test folder is from BSDS500/data/images/test. """
""" the local ../images/test/train folder is from BSDS500/data/images/train. """
parser = argparse.ArgumentParser(description="Demo of CSNet.")
parser.add_argument("--epochs", default=100, type=int, metavar="NUM")
parser.add_argument("--batch_size", default=20, type=int, metavar="SIZE")
parser.add_argument("--block_size", default=32, type=int, metavar="SIZE")
# parser.add_argument("--image_size", default=256, type=int, metavar="SIZE")
parser.add_argument("--test_path", default="../images/test", metavar="PATH")
parser.add_argument("--train_path", default="../images/train", metavar="PATH")
opt = parser.parse_args()


def loader():
    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomCrop(opt.block_size)
        # torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomCrop(opt.block_size)
        # torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_dataset = torchvision.datasets.ImageFolder(opt.test_path, transform=test_transforms)
    train_dataset = torchvision.datasets.ImageFolder(opt.train_path, transform=train_transforms)

    test_data_loader = data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=True)
    train_data_loader = data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    return train_data_loader, test_data_loader


def train(net):
    if not os.path.isfile("./CSNet.pth"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net.to(device)

        train_set, _ = loader()

        criterion = nn.MSELoss()
        optimizer = optimize.Adam(net.parameters())
        scheduler = optimize.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 80], gamma=0.1)

        print("=> Train start.")
        time_start = time.time()

        for epoch in range(opt.epochs):
            for i, (input, name) in enumerate(train_set):
                optimizer.zero_grad()
                output = net(input)
                loss = criterion(output, input)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                use_time = time.time() - time_start

                print("=> epoch: {}, batch: {:.0f}, loss: {:.4f}, lr: {}, time: {:.3f}"
                      .format(epoch + 1, i + 1, loss.item(), optimizer.param_groups[0]["lr"], use_time))
            scheduler.step()
        print("=> Train end.")
        torch.save(net.state_dict(), net.save_path)


def val(net):
    if os.path.isfile("./CSNet.pth"):
        if not os.path.exists("./images"):
            os.makedirs("./images")

        net.load_state_dict(torch.load('./CSNet.pth', map_location='cpu'))
        tensor2image = torchvision.transforms.ToPILImage()
        criterion = nn.MSELoss()
        _, test_set = loader()
        sum_mse = 0

        for idx, (input, name) in enumerate(test_set):
            with torch.no_grad():
                output = net(input)
                sum_mse += criterion(output, input).item()

                batch_tensor = output.cpu().clone()
                list_tensor = batch_tensor.chunk(opt.batch_size, dim=0)
                for i in range(opt.batch_size):
                    tensor = list_tensor[i].squeeze()
                    image = tensor2image(tensor)
                    image.save("./images/{}.jpg".format(idx * 10 + i + 1))
        average_mse = sum_mse / (len(test_set) * opt.batch_size)
        print("=> sum_mse: {:.4f}, average_mse: {:.4f}".format(sum_mse, average_mse))


if __name__ == "__main__":
    cs_net = CSNet()
    train(cs_net)
    val(cs_net)
