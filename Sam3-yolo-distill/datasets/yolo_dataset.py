# -*- coding: utf-8 -*-

"""
YOLOv8-seg dataset for SAM3 distillation

Output:

{
    img:
        Tensor[B,3,H,W]

    cls:
        Tensor[N,1]

    bboxes:
        Tensor[N,4]

    masks:
        Tensor[N,H,W]

    batch_idx:
        Tensor[N]

    prompts:
        List[str]

}

Compatible with:

YOLOv8-seg loss

SAM3 text prompt
"""


import os

import cv2

import torch

from torch.utils.data import Dataset


from configs.classes import CLASS_PROMPTS



class YOLODataset(Dataset):


    def __init__(
        self,
        image_dir,
        label_dir,
        img_size=640
    ):


        self.image_dir = image_dir

        self.label_dir = label_dir

        self.img_size = img_size



        self.images = [

            x for x in os.listdir(image_dir)

            if x.endswith(
                (
                    ".jpg",
                    ".png",
                    ".jpeg"
                )
            )

        ]


        self.images.sort()



    def __len__(self):

        return len(self.images)



    def __getitem__(
        self,
        index
    ):


        image_name = self.images[index]



        image_path = os.path.join(

            self.image_dir,

            image_name

        )



        label_path = os.path.join(

            self.label_dir,

            os.path.splitext(
                image_name
            )[0]
            +
            ".txt"

        )



        # =================================
        # image
        # =================================


        img = cv2.imread(
            image_path
        )


        img = cv2.cvtColor(

            img,

            cv2.COLOR_BGR2RGB

        )



        img = cv2.resize(

            img,

            (
                self.img_size,
                self.img_size
            )

        )



        img = (

            torch.from_numpy(img)

            .permute(
                2,
                0,
                1
            )

            .float()

            /255.0

        )



        # =================================
        # labels
        # =================================


        cls = []

        boxes = []

        masks = []


        #
        # SAM3 prompt
        #

        prompts = []



        if os.path.exists(
            label_path
        ):



            with open(
                label_path,
                "r"
            ) as f:


                lines = f.readlines()



            for line in lines:


                data = list(

                    map(
                        float,
                        line.strip().split()
                    )

                )


                class_id = int(
                    data[0]
                )



                points = data[1:]



                polygon = []



                for i in range(
                    0,
                    len(points),
                    2
                ):


                    x = (

                        points[i]

                        *

                        self.img_size

                    )


                    y = (

                        points[i+1]

                        *

                        self.img_size

                    )


                    polygon.append(

                        [
                            x,
                            y
                        ]

                    )



                polygon = torch.tensor(

                    polygon,

                    dtype=torch.float32

                )



                # =========================
                # bbox
                # =========================


                xmin = polygon[:,0].min()

                ymin = polygon[:,1].min()

                xmax = polygon[:,0].max()

                ymax = polygon[:,1].max()



                cx = (

                    xmin+xmax

                ) / 2



                cy = (

                    ymin+ymax

                ) / 2



                bw = xmax-xmin

                bh = ymax-ymin



                boxes.append(

                    [

                        cx/self.img_size,

                        cy/self.img_size,

                        bw/self.img_size,

                        bh/self.img_size

                    ]

                )



                cls.append(

                    [
                        class_id
                    ]

                )



                # =========================
                # SAM3 text prompt
                # =========================


                if class_id in CLASS_PROMPTS:


                    prompts.append(

                        CLASS_PROMPTS[class_id]

                    )


                else:


                    prompts.append(

                        "object"

                    )



                # =========================
                # mask
                # =========================


                mask = torch.zeros(

                    self.img_size,

                    self.img_size

                )



                pts = polygon.numpy().astype(

                    "int32"

                )



                cv2.fillPoly(

                    mask.numpy(),

                    [
                        pts
                    ],

                    1

                )



                masks.append(
                    mask
                )



        # =================================
        # empty target
        # =================================


        if len(boxes)==0:


            boxes = torch.zeros(

                (0,4),

                dtype=torch.float32

            )


            cls = torch.zeros(

                (0,1),

                dtype=torch.float32

            )


            masks = torch.zeros(

                0,

                self.img_size,

                self.img_size

            )


            prompts = []



        else:


            boxes = torch.tensor(

                boxes,

                dtype=torch.float32

            )


            cls = torch.tensor(

                cls,

                dtype=torch.float32

            )


            masks = torch.stack(

                masks

            )



        return {


            "img":

            img,


            "cls":

            cls,


            "bboxes":

            boxes,


            "masks":

            masks,


            "prompts":

            prompts,


            "batch_idx":

            torch.zeros(

                len(cls),

                dtype=torch.long

            )

        }



    # =====================================
    # collate
    # =====================================


    @staticmethod
    def collate_fn(batch):


        new_batch = {}



        keys = batch[0].keys()



        for key in keys:



            values = [

                x[key]

                for x in batch

            ]



            if key == "img":


                new_batch[key] = torch.stack(

                    values,

                    0

                )



            elif key in [

                "cls",

                "bboxes",

                "masks"

            ]:


                new_batch[key] = torch.cat(

                    values,

                    0

                )



            elif key == "batch_idx":


                batch_idx = []



                for i,v in enumerate(values):


                    batch_idx.append(

                        torch.full_like(

                            v,

                            i

                        )

                    )



                new_batch[key] = torch.cat(

                    batch_idx,

                    0

                )



            elif key == "prompts":


                #
                # list[str]
                #

                prompts = []


                for p in values:


                    prompts.extend(
                        p
                    )


                new_batch[key] = prompts



        return new_batch