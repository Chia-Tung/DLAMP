from collections import defaultdict

import torch


def prediction_postprocess(
    trainer_output: list[list[torch.Tensor]], mapping: dict[int, str]
):
    """
    Perform post-processing on the trainer output predictions by combining all the batches.

    Parameters:
        trainer_output (list[list[torch.Tensor]]): The output predictions from the trainer.
            Shape: [epochs][num_product_type][B, lv, h, w, c]
        mapping (dict[int, str]): A mapping dictionary representing the order of trainer_output.
            The structure is like:
            {
                "input_upper": 0,
                "input_sruface": 1,
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
        predictions[k] = torch.cat(v, dim=0)  # {"input_upper": (B, lv, h, w, c)...}

    return predictions
