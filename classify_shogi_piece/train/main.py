import os
from typing import Any, Final, List, TypedDict

import torch
import torchvision
from torch import nn, optim
from torchvision import datasets, models, transforms

DATASET_DIR: Final[str] = "../../data/dataset/"
MODEL_PATH: Final[str] = "../../models/model.pth"
CLASSES: Final[List[str]] = sorted(
    [
        "-01",
        "-02",
        "-03",
        "-04",
        "-05",
        "-06",
        "-07",
        "-08",
        "-09",
        "-10",
        "-11",
        "-12",
        "-13",
        "-14",
        "00",
        "01",
        "02",
        "03",
        "04",
        "05",
        "06",
        "07",
        "08",
        "09",
        "10",
        "11",
        "12",
        "13",
        "14",
    ]
)

DeviceType: Final[Any] = torch._C.device
TransformType: Final[Any] = transforms.Compose
ModelType: Final[Any] = torchvision.models.densenet161
DataSetType: Final[Any] = datasets.folder.ImageFolder
DataLoadersType: Final[Any] = TypedDict(
    "DataLoadersType",
    {
        "train": torch.utils.data.DataLoader,
        "test": torch.utils.data.DataLoader,
    },
)
CriterionType: Final[Any] = nn.CrossEntropyLoss
OptimizerType: Final[Any] = optim.SGD
SchedulerType: Final[Any] = optim.lr_scheduler.StepLR


def get_data_loaders(dataset_dir: str, transform: Any) -> DataLoadersType:
    my_datasets: dict[str, DataSetType] = {
        "train": datasets.ImageFolder(
            root=os.path.join(dataset_dir, "train"), transform=transform
        ),
        "test": datasets.ImageFolder(
            root=os.path.join(dataset_dir, "test"), transform=transform
        ),
    }

    data_loaders: DataLoadersType = {
        "train": torch.utils.data.DataLoader(
            my_datasets["train"], batch_size=30, shuffle=True, num_workers=2
        ),
        "test": torch.utils.data.DataLoader(
            my_datasets["test"], batch_size=10, shuffle=False, num_workers=1
        ),
    }
    return data_loaders


def get_model(
    pretrained: bool, class_quantity: int, device: DeviceType
) -> ModelType:
    # densenetを採用
    model = models.densenet161(pretrained=pretrained)
    # 最終層のラベル数を調整
    model.classifier = nn.Linear(model.classifier.in_features, class_quantity)
    # GPUに乗せる
    densenet = model.to(device)
    return densenet


def train_model(
    model: ModelType,
    data_loaders: DataLoadersType,
    criterion: CriterionType,
    optimizer: OptimizerType,
    scheduler: SchedulerType,
    device: DeviceType,
    num_epochs: int,
    target_accuracy: float = 99.5,
    running_loss_step: int = 10,
) -> None:
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1} / {num_epochs}")
        print("----------")

        # バッジ学習10回毎の損失記録用
        running_loss_sum: float = 0
        # エポックの学習時損失記録用
        train_loss_sum: float = 0
        # エポックの評価時損失記録用
        test_loss_sum: float = 0

        # 学習回数の記録用
        train_count: int = 0
        test_count: int = 0

        class_correct = {x: 0 for x in CLASSES}
        class_total = {x: 0 for x in CLASSES}

        model.train()
        for i, (inputs, labels) in enumerate(data_loaders["train"]):
            inputs = inputs.to(device)
            labels = labels.to(device)
            # 勾配の初期化
            optimizer.zero_grad()
            # 予測
            outputs = model(inputs)
            # 損失算出
            loss = criterion(outputs, labels)
            # 逆伝播
            loss.backward()
            # 勾配の更新
            optimizer.step()

            # 損失加算
            running_loss_sum += loss.item()
            if (i + 1) % running_loss_step == 0:
                running_loss_ave: float = round(
                    running_loss_sum / running_loss_step, 6
                )
                print(f"{i+1}\trunning_loss_ave:\t{running_loss_ave}")
                running_loss_sum = 0

            # 学習時の損失を追加
            train_loss_sum += loss.item()
            train_count += 1

        scheduler.step()

        # モデルの評価
        model.eval()
        for i, (inputs, labels) in enumerate(data_loaders["test"]):
            inputs = inputs.to(device)
            labels = labels.to(device)
            # 予測
            outputs = model(inputs)
            # 損失算出
            loss = criterion(outputs, labels)
            # 損失加算
            test_loss_sum += loss.item()
            # 正解数の算出
            _, predicted = torch.max(outputs, 1)
            is_corrects = (predicted == labels).squeeze()

            for j, is_correct in enumerate(is_corrects):
                label = CLASSES[labels[j]]
                class_correct[label] += is_correct.item()
                class_total[label] += 1

            test_count += 1

        print(f"train_loss_ave:\t{train_loss_sum / train_count}")
        print(f"test_loss_ave:\t{test_loss_sum / test_count}")
        for label in CLASSES:
            label_accuracy = class_correct[label] / class_total[label] * 100
            print(f"{label}:\t{label_accuracy} %")

        accuracy = (
            sum(class_correct.values()) / sum(class_total.values()) * 100
        )
        print(f"ALL:\t{accuracy} %")
        if accuracy > target_accuracy:
            break


if __name__ == "__main__":
    device: DeviceType = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    # 前処理の定義
    transform: TransformType = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    # データ読み込み用イテレータを定義
    data_loaders: DataLoadersType = get_data_loaders(
        dataset_dir=DATASET_DIR, transform=transform
    )

    # 学習モデル
    model: ModelType = get_model(
        pretrained=True, class_quantity=len(CLASSES), device=device
    )

    # 損失関数
    criterion = nn.CrossEntropyLoss()
    # 最適化関数
    # 最適化関数
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # 10エポックごとに学習率を1/2に低下
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    train_model(
        model=model,
        data_loaders=data_loaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=10,
    )

    torch.save(model.state_dict(), MODEL_PATH)
