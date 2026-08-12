# -*- coding:utf-8 -*-


import torch

from torch.utils.data import DataLoader


from train_distill import create_trainer


from datasets.yolo_dataset import YOLODataset



IMAGE_DIR = (
"/data/ultralytics/"
"Sam3-yolo-distill/tests/data/images/train"
)


LABEL_DIR = (
"/data/ultralytics/"
"Sam3-yolo-distill/tests/data/labels/train"
)



def main():


    trainer = create_trainer()



    dataset = YOLODataset(

        image_dir=IMAGE_DIR,

        label_dir=LABEL_DIR,

        img_size=640

    )



    loader = DataLoader(

        dataset,

        batch_size=8,

        shuffle=True,

        num_workers=4,

        pin_memory=True,

        collate_fn=dataset.collate_fn

    )



    epochs=100



    for epoch in range(epochs):


        print(
            "epoch:",
            epoch
        )


        for step,batch in enumerate(loader):


            # ======================
            # move cuda
            # ======================

            for k,v in batch.items():

                if torch.is_tensor(v):

                    batch[k]=v.cuda(
                        non_blocking=True
                    )



            result = trainer.train_step(
                batch
            )


            if step % 10 == 0:

                print(
                    epoch,
                    step,
                    result
                )



if __name__=="__main__":

    main()