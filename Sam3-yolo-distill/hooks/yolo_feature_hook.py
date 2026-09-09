# -*- coding: utf-8 -*-

"""
YOLO Feature Hook

Extract YOLOv8-seg P3/P4/P5 feature

Student:
    YOLOv8-seg

Output:

p3
p4
p5

"""


import torch



class YOLOFeatureHook:


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
        Segment head input:

        inputs:

        (
          [
            P3,
            P4,
            P5
          ]
        )

        """



        if not isinstance(
            inputs,
            tuple
        ):

            raise RuntimeError(
                "Unexpected YOLO head input"
            )


        pyramid = inputs[0]



        if not isinstance(
            pyramid,
            list
        ):

            raise RuntimeError(
                "YOLO feature should be list"
            )



        if len(pyramid)!=3:

            raise RuntimeError(
                f"Expected 3 YOLO features, got {len(pyramid)}"
            )



        self.features={


            "p3":pyramid[0],


            "p4":pyramid[1],


            "p5":pyramid[2]


        }




    def register(self):

        """
        register hook

        """



        # YOLO Segment head

        head = self.model.model[-1]



        self.handle = (

            head.register_forward_hook(
                self._hook_fn
            )

        )


        return self



    def remove(self):


        if self.handle:


            self.handle.remove()

            self.handle=None




    def get_features(self):


        if self.features is None:


            raise RuntimeError(
                "YOLO features not collected. "
                "Run forward first."
            )


        return self.features



    def clear(self):


        self.features=None