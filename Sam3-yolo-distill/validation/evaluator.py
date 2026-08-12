# -*- coding: utf-8 -*-

"""
Validation evaluator

SAM3 -> YOLOv8-seg Distillation

Only evaluate

No backward

"""


import torch



class DistillEvaluator:


    def __init__(

        self,

        teacher,

        student,

        adapters,

        feature_loss,

        yolo_hook,

        device="cuda"

    ):


        self.teacher = teacher

        self.student = student

        self.adapters = adapters

        self.feature_loss = feature_loss

        self.yolo_hook = yolo_hook

        self.device = device



        self.teacher.eval()

        self.student.eval()


        for adapter in self.adapters:

            adapter.eval()



    # ==========================================
    # evaluate
    # ==========================================


    @torch.no_grad()
    def evaluate(

        self,

        loader

    ):



        total_loss = 0

        total_yolo = 0

        total_feature = 0


        count = 0



        for batch in loader:



            # --------------------------
            # cuda
            # --------------------------


            for k,v in batch.items():


                if torch.is_tensor(v):

                    batch[k] = v.to(

                        self.device,

                        non_blocking=True

                    )



            images = batch["img"]



            # --------------------------
            # SAM3 feature
            # --------------------------


            teacher_features = self.teacher(

                images

            )


            sam3_features = [

                teacher_features[1],

                teacher_features[2],

                teacher_features[3]

            ]



            # --------------------------
            # YOLO forward
            # --------------------------


            self.yolo_hook.clear()


            output = self.student(

                images

            )



            student_features = (

                self.yolo_hook.get_features()

            )



            # --------------------------
            # adapter
            # --------------------------


            adapted_features=[]



            for i,feature in enumerate(

                student_features

            ):


                y = self.adapters[i](

                    feature,

                    target_size=
                    sam3_features[i].shape[-2:]

                )


                adapted_features.append(y)



            # --------------------------
            # feature loss
            # --------------------------


            loss_feature = self.feature_loss(

                adapted_features,

                sam3_features

            )



            # --------------------------
            # yolo loss
            # --------------------------


            yolo_result = self.student.loss(

                batch

            )


            if isinstance(

                yolo_result,

                tuple

            ):

                loss_yolo = yolo_result[0]

            else:

                loss_yolo = yolo_result



            if loss_yolo.ndim > 0:

                loss_yolo = loss_yolo.sum()



            loss = (

                loss_yolo

                +

                loss_feature

            )



            total_loss += loss.item()

            total_yolo += loss_yolo.item()

            total_feature += loss_feature.item()


            count += 1



        return {


            "val_loss":

                total_loss / count,


            "val_loss_yolo":

                total_yolo / count,


            "val_loss_feature":

                total_feature / count

        }