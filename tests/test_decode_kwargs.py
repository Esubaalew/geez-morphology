"""Checkpoint selection and evaluate_gold share the same decode default (beam-2)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import train_byt5_morphology as T  # noqa: E402


class _Cuda:
    @staticmethod
    def is_available():
        return False


class _BackendsNoMps:
    pass


class _TorchNoMps:
    cuda = _Cuda
    backends = _BackendsNoMps()


class _MpsOk:
    @staticmethod
    def is_available():
        return True


class _TorchMps:
    cuda = _Cuda

    class backends:
        mps = _MpsOk()


def test_decode_default_matches_gold_scoring():
    assert T.DEFAULT_NUM_BEAMS == 2
    kw = T.decode_generate_kwargs()
    assert kw["num_beams"] == 2 and kw["do_sample"] is False
    assert kw["max_length"] == T.DEFAULT_GENERATE_MAX_LENGTH == 128
    sel = T.decode_generate_kwargs(max_new_tokens=T.ROOT_ACCURACY_MAX_NEW_TOKENS)
    assert sel["num_beams"] == 2 and sel["max_new_tokens"] == T.ROOT_ACCURACY_MAX_NEW_TOKENS
    assert "max_length" not in sel
    greedy = T.decode_generate_kwargs(num_beams=1, max_new_tokens=32)
    assert greedy["num_beams"] == 1


def test_parse_args_num_beams_default():
    assert T.parse_args([]).num_beams == T.DEFAULT_NUM_BEAMS
    assert T.parse_args(["--num_beams", "1"]).num_beams == 1


def test_pick_device_without_mps_attr():
    assert T.pick_device("auto", torch_module=_TorchNoMps) == "cpu"
    assert T.pick_device("cuda", torch_module=_TorchNoMps) == "cuda"
    assert T.pick_device("auto", torch_module=_TorchMps) == "mps"
