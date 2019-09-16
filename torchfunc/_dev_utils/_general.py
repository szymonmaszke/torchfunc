import multiprocessing
import platform
import typing

import torch


def _general_info():
    return "\n".join(
        [
            f"Python version: {platform.python_version()}",
            f"Python implementation: {platform.python_implementation()}",
            f"Python compiler: {platform.python_compiler()}",
            f"PyTorch version: {torch.__version__}",
            f"System: {platform.system() or 'Unable to determine'}, version: {platform.release() or 'Unable to determine'}",
            f"Processor: {platform.processor() or 'Unable to determine'}",
            f"Number of CPUs: {multiprocessing.cpu_count()}",
        ]
    )


def _cuda_info():
    def _cuda_devices_formatting(
        info_function: typing.Callable,
        formatting_function: typing.Callable = None,
        mapping_function: typing.Callable = None,
    ):
        def _setup_default(function):
            return (lambda arg: arg) if function is None else function

        formatting_function = _setup_default(formatting_function)
        mapping_function = _setup_default(mapping_function)

        return " | ".join(
            mapping_function(
                [
                    formatting_function(info_function(i))
                    for i in range(torch.cuda.device_count())
                ]
            )
        )

    def _device_properties(attribute):
        return _cuda_devices_formatting(
            lambda i: getattr(torch.cuda.get_device_properties(i), attribute),
            mapping_function=lambda in_bytes: map(str, in_bytes),
        )

    return "\n".join(
        [
            "Available CUDA devices count: {}".format(torch.cuda.device_count()),
            "CUDA devices names: {}".format(
                _cuda_devices_formatting(torch.cuda.get_device_name)
            ),
            "Major.Minor CUDA capabilities of devices: {}".format(
                _cuda_devices_formatting(
                    torch.cuda.get_device_capability,
                    formatting_function=lambda capabilities: ".".join(
                        map(str, capabilities)
                    ),
                )
            ),
            "Device total memory (bytes): {}".format(
                _device_properties("total_memory")
            ),
            "Device multiprocessor count: {}".format(
                _device_properties("multi_processor_count")
            ),
        ]
    )
