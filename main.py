import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from infer import convert_model, test
from train import (
    DenseNet121,
    GameDataset,
    create_dataset,
    test_transforms,
    train,
    train_transforms,
    val,
)


def main_train(epochs: int = 10, path_dataset: str = "./test_data/test.txt"):
    df, classes = create_dataset(path=path_dataset)
    scale_split = 88
    train_df = df[:scale_split]
    test_df = df[scale_split:].reset_index()

    train_dataset = GameDataset(df=train_df, transform=train_transforms)
    test_dataset = GameDataset(df=test_df, transform=test_transforms)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=16,
        pin_memory=True,
        drop_last=False,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        pin_memory=True,
    )

    model = DenseNet121(classCount=33, isTrained=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for e in range(epochs):
        train(model, optimizer, criterion, e, train_loader=train_loader)
        precision, recall = val(model, optimizer, criterion, e, test_loader=test_loader)
        print(f"Precision validation: {precision},Recall validation: {recall}")

    print("Finished Training")
    PATH = "./save_model/hero_model.pth"
    torch.save(model.state_dict(), PATH)


def main_infer(checkpoint, checkpoint_jit, folder_img, save_file):
    convert_model(checkpoint)
    test(checkpoint=checkpoint_jit, folder_img=folder_img, save_file=save_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Hero Name Recognition")
    subparsers = parser.add_subparsers(help="Actions", dest="action")

    train_parser = subparsers.add_parser("train", help="Start training process")
    train_parser.add_argument(
        "-e", "--epochs", default=10, help="Numbers loop traning", type=int
    )
    train_parser.add_argument(
        "-p", "--path_dataset", default="./test_data/test.txt", help="Datasets"
    )

    export_parser = subparsers.add_parser(
        "export", help="Export model to TorchScript format"
    )
    export_parser.add_argument(
        "-c",
        "--convert_model",
        default="./save_model/hero_model.pth",
        help="Export model to TorchScript format",
    )

    infer_parser = subparsers.add_parser(
        "infer",
        help="Infer data with TorchScript model, export report if label(s) provided",
    )
    infer_parser.add_argument(
        "-m",
        "--torchjit_ck",
        default="./save_model/model_hero_jit.pt",
        help="torchscript model",
    )
    infer_parser.add_argument(
        "-f", "--folder_img", default="./test_data/test_images", help="Folder Image"
    )
    infer_parser.add_argument(
        "-s", "--save_file", default="./output.txt", help="save results"
    )

    opts = vars(parser.parse_args())

    if opts.get("action") == "train":
        main_train(epochs=opts["epochs"], path_dataset=opts["path_dataset"])
    if opts.get("action") == "export":
        convert_model(checkpoint=opts["convert_model"])
    if opts.get("action") == "infer":
        test(
            file_convert_model=opts["torchjit_ck"],
            folder_img=opts["folder_img"],
            save_file=opts["save_file"],
        )
