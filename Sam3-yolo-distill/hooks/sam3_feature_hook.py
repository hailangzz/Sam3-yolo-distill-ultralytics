# -*- coding: utf-8 -*-

"""
SAM3 Feature Hook

Extract SAM3 vision features for distillation.

Teacher:
    SAM3 vision backbone

Output:
    P3
    P4
    P5

Example:

P3:
[B,256,H/8,W/8]

P4:
[B,256,H/16,W/16]

P5:
[B,256,H/32,W/32]

"""


import torch


class SAM3FeatureHook:
    """
    Hook SAM3 vision backbone output

    Extract:
        P3
        P4
        P5

    """


    def __init__(
        self,
        model
    ):

        self.model = model


        self.features = None


        self.handle = None



    def _hook_fn(
        self,
        module,
        inputs,
        outputs
    ):

        """
        outputs:

        tuple(
            level0,
            level1,
            level2,
            level3
        )

        each level:

        [
          P2,
          P3,
          P4,
          P5
        ]

        """

        if not isinstance(outputs, tuple):

            raise RuntimeError(
                "Unexpected SAM3 vision output type"
            )


        # 使用第一个level
        # 经过验证4个level结构一致
        pyramid = outputs[0]


        if not isinstance(
            pyramid,
            list
        ):

            raise RuntimeError(
                "Unexpected SAM3 pyramid format"
            )


        if len(pyramid) != 4:

            raise RuntimeError(
                f"Expected 4 feature levels, got {len(pyramid)}"
            )


        # SAM3 pyramid:
        #
        # 0 P2
        # 1 P3
        # 2 P4
        # 3 P5


        self.features = {

            "p3": pyramid[1],

            "p4": pyramid[2],

            "p5": pyramid[3],

        }



    def register(self):

        """
        register forward hook
        """


        self.handle = (
            self.model
            .backbone
            .vision_backbone
            .register_forward_hook(
                self._hook_fn
            )
        )


        return self



    def remove(self):

        """
        remove hook
        """

        if self.handle:

            self.handle.remove()

            self.handle = None



    def get_features(self):

        """
        return:

        {
            p3: Tensor,
            p4: Tensor,
            p5: Tensor
        }

        """


        if self.features is None:

            raise RuntimeError(
                "SAM3 features not collected. "
                "Run forward first."
            )


        return self.features



    def clear(self):

        self.features=None