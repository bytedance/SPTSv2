# ------------------------------------------------------------------------
# Copyright (2023) Bytedance Ltd. and/or its affiliates
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
from .sptsv2 import build


def build_model(args):
    return build(args)
