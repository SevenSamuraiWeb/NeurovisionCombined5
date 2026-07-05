"""Package init. Installs the torchvision/basicsr compatibility shim before any
service module can import basicsr: basicsr 1.4.2 references
torchvision.transforms.functional_tensor, removed in torchvision 0.16+.
"""
import sys
import types

if "torchvision.transforms.functional_tensor" not in sys.modules:
    import torchvision.transforms.functional as _tvf

    _shim = types.ModuleType("torchvision.transforms.functional_tensor")
    _shim.rgb_to_grayscale = _tvf.rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = _shim
