from datasets.yolo_dataset import YOLODataset


dataset = YOLODataset(

    "/home/chenkejing/database/test/data/images/train",

    "/home/chenkejing/database/test/data/labels/train",

    640

)


sample = dataset[0]


print(sample.keys())

print(sample["prompts"])