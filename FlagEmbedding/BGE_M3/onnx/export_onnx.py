import argparse
import copy
from collections import OrderedDict
from typing import Dict

from optimum.exporters.onnx import onnx_export_from_model
from optimum.exporters.onnx.model_configs import XLMRobertaOnnxConfig
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers_fake_module import BGEM3InferenceModel


class BGEM3OnnxConfig(XLMRobertaOnnxConfig):
    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        """
        Dict containing the axis definition of the output tensors to provide to the model.
        Returns:
            `Dict[str, Dict[int, str]]`: A mapping of each output name to a mapping of axis position to the axes symbolic name.
        """
        return copy.deepcopy(
            OrderedDict(
                {
                    "dense_vecs": {0: "batch_size", 1: "embedding"},
                    "sparse_vecs": {0: "batch_size", 1: "token", 2: "weight"},
                    "colbert_vecs": {0: "batch_size", 1: "token", 2: "embedding"},
                }
            )
        )


def main(
    input: str,
    output: str,
    opset: int,
    device: str,
    optimize: str,
    atol: str,
    quantize: bool,
):
    model = BGEM3InferenceModel(model_name=input)
    model = model.to(device)
    bgem3_onnx_config = BGEM3OnnxConfig(model.config)
    onnx_export_from_model(
        model,
        output=output,
        task="feature-extraction",
        custom_onnx_configs={"model": bgem3_onnx_config},
        opset=opset,
        optimize=optimize,
        atol=atol,
        device=device,
    )
    if quantize:
        quantizer = ORTQuantizer.from_pretrained(output, file_name="model.onnx")
        qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=True)
        quantizer.quantize(save_dir=output, quantization_config=qconfig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        help="Input model path",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path indicating the directory where to store the generated ONNX model.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=None,
        help="If specified, ONNX opset version to export the model with. Otherwise, the default opset for the given model architecture will be used.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help='The device to use to do the export. Defaults to "cpu".',
    )
    parser.add_argument(
        "--optimize",
        type=str,
        default=None,
        choices=["O1", "O2", "O3", "O4"],
        help=(
            "Allows to run ONNX Runtime optimizations directly during the export. Some of these optimizations are specific to ONNX Runtime, and the resulting ONNX will not be usable with other runtime as OpenVINO or TensorRT. Possible options:\n"
            "    - O1: Basic general optimizations\n"
            "    - O2: Basic and extended general optimizations, transformers-specific fusions\n"
            "    - O3: Same as O2 with GELU approximation\n"
            "    - O4: Same as O3 with mixed precision (fp16, GPU-only, requires `--device cuda`)"
        ),
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=None,
        help="If specified, the absolute difference tolerance when validating the model. Otherwise, the default atol for the model will be used.",
    )

    parser.add_argument(
        "--quantize",
        type=bool,
        default=False,
        help="If specified, the model will be quantized using ONNX Runtime quantization",
    )
    args = parser.parse_args()

    main(
        args.input,
        args.output,
        args.opset,
        args.device,
        args.optimize,
        args.atol,
        args.quantize,
    )
