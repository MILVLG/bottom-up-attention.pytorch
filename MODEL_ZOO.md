# Model Zoo

| |bacbone|objects mAP@0.5|objects weighted mAP@0.5|
-|-|-|-
|[Faster R-CNN]()|ResNet-101|10.2%|15.1%|

Note that mAP is relatively low because many classes overlap (e.g. person / man / guy), some classes can't be precisely located (e.g. street, field) and separate classes exist for singular and plural objects (e.g. person / people). We focus on performance in downstream tasks (e.g. image captioning, VQA) rather than detection performance.