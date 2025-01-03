import time
from datetime import datetime

import hydra
import onnxruntime as ort
import psutil
from omegaconf import DictConfig, OmegaConf

from src.managers import DataManager
from src.standardization import destandardization
from src.utils import DataCompose

"""
This is a sample code for quickly inference onnx model.
"""


@hydra.main(version_base=None, config_path="../config", config_name="predict")
def main(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, True)

    # prepare data
    eval_cases = [datetime(2022, 9, 11)]
    data_list = DataCompose.from_config(cfg.data.train_data)
    data_manager = DataManager(data_list, eval_cases, **cfg.data, **cfg.lightning)
    data_manager.setup("predict")

    # sample data
    data_loader = data_manager.predict_dataloader()
    inp_data, oup_data = next(iter(data_loader))

    # onnxruntime settings
    assert "CUDAExecutionProvider" in ort.get_available_providers()
    print(f"ort device: {ort.get_device()}")

    # An issue about onnxruntime for cuda12.x
    # ref: https://github.com/microsoft/onnxruntime/issues/8313#issuecomment-1486097717
    _default_session_options = ort.capi._pybind_state.get_default_session_options()

    def get_default_session_options_new():
        _default_session_options.inter_op_num_threads = 1
        _default_session_options.intra_op_num_threads = 1
        return _default_session_options

    ort.capi._pybind_state.get_default_session_options = get_default_session_options_new

    # inference
    onnx_filename = cfg.inference.onnx_path
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = psutil.cpu_count(logical=True)  # faster
    ort_session = ort.InferenceSession(
        onnx_filename,
        sess_options,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    ort_inputs = {
        ort_session.get_inputs()[0].name: inp_data["upper_air"].cpu().numpy(),
        ort_session.get_inputs()[1].name: inp_data["surface"].cpu().numpy(),
    }
    start = time.time()
    pred_upper, pred_surface = ort_session.run(None, ort_inputs)
    pred_upper = destandardization(pred_upper)
    pred_surface = destandardization(pred_surface)
    print(type(pred_upper), pred_upper.shape, pred_surface.shape)
    print(f"execution time: {time.time() - start:.5f} sec")


if __name__ == "__main__":
    main()
