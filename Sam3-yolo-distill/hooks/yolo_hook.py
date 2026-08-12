# -*- coding: utf-8 -*-

"""
YOLOv8 Feature Hook

P3/P4/P5 extractor
"""


class YOLOFeatureHook:



    def __init__(
        self,
        model,
        layers=(15,18,21)
    ):


        self.model=model

        self.layers=layers

        self.features={}

        self.handles=[]


        self.register()



    def register(self):


        for idx in self.layers:


            module = self.model.model[idx]


            handle = module.register_forward_hook(

                self.make_hook(idx)

            )


            self.handles.append(handle)



    def make_hook(
        self,
        idx
    ):


        def hook(
            module,
            input,
            output
        ):


            self.features[idx]=output



        return hook




    def clear(self):

        self.features.clear()



    def get_features(self):


        return [

            self.features[i]

            for i in self.layers

        ]



    def remove(self):


        for h in self.handles:

            h.remove()


        self.handles=[]
