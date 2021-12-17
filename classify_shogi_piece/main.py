from dataclasses import InitVar, dataclass, field
from typing import Any, Final, List

import numpy as np
import numpy.typing as npt
import torch
import torchvision
from torch import nn
from torchvision import models, transforms

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

IntegerArrayType: Final[Any] = npt.NDArray[np.int_]
DeviceType: Final[Any] = torch._C.device
TransformType: Final[Any] = transforms.Compose
ModelType: Final[Any] = torchvision.models.densenet161
TensorType: Final[Any] = torch.Tensor


@dataclass
class ShogiModel:
    device: DeviceType = field(init=False)
    transform: TransformType = field(init=False)
    model: ModelType = field(init=False)
    model_path: InitVar[str] = field(default="../models/model.pth")

    def __post_init__(self, model_path):
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.model = models.densenet161()
        self.model.classifier = nn.Linear(
            self.model.classifier.in_features, len(CLASSES)
        )
        self.model = self.model.to(device=self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )

    def predict(self, images: IntegerArrayType) -> IntegerArrayType:
        results: IntegerArrayType = np.array([])
        self.model.eval()
        with torch.no_grad():
            for image in images:
                inputs: TensorType = (
                    self.transform(image).unsqueeze(0).to(self.device)
                )
                outputs: TensorType = self.model(inputs)
                predicted: TensorType = torch.max(outputs, 1)[1]
                results = np.append(results, int(CLASSES[predicted]))
        return results.reshape(9, 9).astype(np.int32)
