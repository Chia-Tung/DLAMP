from collections import defaultdict
from functools import partial, wraps
from typing import Callable

import numpy as np
import onnxruntime as ort
import torch
import yaml
from torch import nn

from src.models.builders.pangu_builder import PanguBuilder
from src.standardization import destandardization
from src.utils import DataCompose


def prediction_postprocess(
    trainer_output: list[list[torch.Tensor]], mapping: dict[int, str]
) -> defaultdict[str, torch.Tensor]:
    """
    Perform post-processing on the trainer output predictions by combining all the batches.

    Parameters:
        trainer_output (list[list[torch.Tensor]]): The output predictions from the trainer.
            Shape: [epochs][num_product_type][B, lv, h, w, c]
        mapping (dict[int, str]): A mapping dictionary representing the order of trainer_output.
            The structure is like:
            {
                "input_upper": 0,
                "input_surface": 1,
                "target_upper": 2,
                "target_surface": 3,
                "output_upper": 4,
                "output_surface": 5,
            }
            For more details, please refer to `XXXLightningModule.get_pred_mapping()`

    Returns:
        defaultdict: Processed predictions in shape: {product_type: (B, lv, h, w, c)...}
    """
    predictions = defaultdict(list)
    for epoch_id in range(len(trainer_output)):
        for key, value in mapping.items():
            predictions[key].append(trainer_output[epoch_id][value])

    for k, v in predictions.items():
        if "output" in k:
            tmp = [destandardization(ele.cpu().numpy()) for ele in v]
            tmp = np.stack(tmp, axis=0)  # {"output_upper": (B, Seq, lv, h, w, c)}
        else:
            tmp = torch.cat(v, dim=0)  # {"input_upper": (B, lv, h, w, c)...}
            tmp = destandardization(tmp.cpu().numpy())
        predictions[k] = torch.from_numpy(tmp)

    return predictions


def ort_instance_decorator(func):
    onnx_path = None
    gpu_id = None

    @wraps(func)
    def wrapper(**kwargs) -> ort.InferenceSession | partial:
        nonlocal onnx_path, gpu_id
        if "onnx_path" in kwargs:
            onnx_path = kwargs["onnx_path"]
        if "gpu_id" in kwargs:
            gpu_id = kwargs["gpu_id"]

        if onnx_path is not None and gpu_id is not None:
            return func(onnx_path=onnx_path, gpu_id=gpu_id)
        elif onnx_path is not None:
            return partial(func, onnx_path=onnx_path)
        elif gpu_id is not None:
            return partial(func, gpu_id=gpu_id)
        else:
            raise RuntimeError(
                "onnx_path and gpu_id are both None. Please specify onnx_path and gpu_id."
            )

    return wrapper


@ort_instance_decorator
def init_ort_instance(gpu_id: int, onnx_path: str) -> ort.InferenceSession:
    assert "CUDAExecutionProvider" in ort.get_available_providers()

    # An issue about onnxruntime for cuda12.x
    # ref: https://github.com/microsoft/onnxruntime/issues/8313#issuecomment-1486097717
    _default_session_options = ort.capi._pybind_state.get_default_session_options()

    def get_default_session_options_new():
        _default_session_options.inter_op_num_threads = 1
        _default_session_options.intra_op_num_threads = 1
        return _default_session_options

    ort.capi._pybind_state.get_default_session_options = get_default_session_options_new

    return ort.InferenceSession(
        onnx_path,
        providers=[
            (
                "CUDAExecutionProvider",
                {
                    "device_id": gpu_id,
                },
            ),
            "CPUExecutionProvider",
        ],
    )


def load_pangu_model(
    ckpt_path: str, data_list: list[DataCompose], image_shape: list[int, int]
) -> Callable[[torch.device], nn.Module]:
    with open("./config/model/pangu_rwrf.yaml") as stream:
        cfg_model = yaml.safe_load(stream)
    with open("./config/lightning/pangu_rwrf.yaml") as stream:
        cfg_lightning = yaml.safe_load(stream)

    # build model
    pangu_builder = PanguBuilder(
        "dummy", data_list, image_shape=image_shape, **cfg_model, **cfg_lightning
    )
    model = pangu_builder._backbone_model()

    # load weights from checkpoint
    ckpt = torch.load(ckpt_path, weights_only=False)
    state_dict = {
        k.replace("backbone_model.", ""): v for k, v in ckpt["state_dict"].items()
    }
    model.load_state_dict(state_dict)

    return lambda device: model.to(device)
