# Project title: Multi-disease diagnosis of chest X-ray in real-world data

# Abstract:
The COVID-19 pandemic has led to massive demands for Chest X-ray diagnosis, dramatically overloading frontier radiologists and promoting the development of artificial intelligence (AI)-based diagnostic models to address this situation. However, the current AI models were still insufficient in multi-disease diagnosis for the real-world data, which showed a long-tail distribution. To tackle this issue, a long-tail public dataset, “ChestX-ray14”, which involved fourteen (14) disease labels, was randomly divided into train, val, and test sets with ratios of 0.7, 0.1, and 0.2. Two pretrained state-of-the-art networks, EfficientNet-b5 and CoAtNet-0-rw, were chosen as the backbones. After the fully-connected layer, a final layer of 14 sigmoid activation units was added to output each disease’s probability and prediction. To achieve better adaptive learning, a novel loss (Lours) was designed, which coalesced reweighting and contribution reduction of head samples. For comparison, a pretrained ResNet50 network with weighted binary cross-entropy loss (LWBCE) was used as a baseline, which showed the best performance in a previous study. The overall and individual areas under the receiver operating curve (AUROC) for each disease label were evaluated and compared among different models. Group-score-weighted class activation mapping (Group-CAM) is applied for visual interpretations. As a result, the pretrained CoAtNet-0-rw+ Lours showed the best overall AUROC of 0.842, higher than the baseline (AUROC: 0.811). Group-CAM presented that the model could pay the proper attention to lesions for most disease labels (e.g., atelectasis, edema, effusion, etc.) but wrong attention for the other labels, such as pneumothorax; meanwhile, mislabeling of the dataset was found. In conclusion, this study achieved an improved performance of multi-disease AI diagnosis. However, the finite sample size of disease cases and sorts of inaccurate labeling probably limited the diagnostic capability of the model. 





## 环境
 - cuda 11.3
 - python 3.9
 - pytorch 1.11

## 使用
1. 准备数据
请根据实际路径修改脚本里面的路径
```
sh ./scrypt/NIH_prepare_data.sh
sh ./scrypt/data_split.sh
or use the code in Train.ipynb
```

2. 修改标签
 - NIH中一个数据可能对应多个标签，在[1]中生成的数据yaml文件中，标签项类型为list型
 - 请修改data_loader.py行中，对```if isinstance(label, list)```中多标签数据的标签指定方式
 ```
    def __getitem__(self, index):
        data = self.data[index]
        image = PIL.Image.open(os.path.join(self.root_dir, data["path"]))
        label = data["label"]

        if isinstance(label, list):
            label = label[0]

        if label in self.classes:
            label = self.classes.index(label)
        file_name = data["file_name"]

        if self.transform is not None:
            image = self.transform(image)
        
        return image, label, file_name
 ```

3. 修改配置文件./config/NIH.yaml
4. 运行训练脚本
```
sh ./scrypt/NIH_train.sh
# 指定GPU
CUDA_VISIBLE_DEVICES=0,1 sh ./scrypt/NIH_train.sh
# 分布式训练
CUDA_VISIBLE_DEVICES=0.1 sh scrypt/NIH_train_distributed.sh
or use the code in Train.ipynb
```
5. 根据需要修改evaluate.py中ROC_AUC计算方法
```
#see more info: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
roc_auc = metrics.roc_auc_score(total_label.detach().cpu().numpy(), total_label_prob.detach().cpu().numpy(), multi_class='ovr')
```
6. 运行评估脚本
```
sh ./scrypt/NIH_evaluate.yaml
```
## 其他
1. 如果需要修改yaml为json，可以修改utils.py中load_yaml()和save_yaml()函数，其他代码均使用这两个函数读取yaml获取dict

## 更新说明

### 2022/10/11
1. 增加了[timm库](https://github.com/rwightman/pytorch-image-models)，使用其中提供的CoAtNet代码与预训练权重。
目前更新的预训练权重参见timm.model.maxxvit:73
2. 增加了WASL loss
用于平衡正负样本的问题。使用时调节```gamma_neg```参数，查看正样本准确率。
这里我的方法是控制训练初期的正样本准确率在50%左右。
3. 增加了一些常用的trick
   - ema: 对模型参数进行指数平滑
   - cutout: 数据增强
   - 混合精度加速: 加速前向传播
   - one cycle scheduler: 一种提供warmup的scheduler


### 2022/10/18
1. 增加saliency map输出功能[gradcam\group_cam\score_cam](https://github.com/wofmanaf/Group-CAM)
2. gradcampp由于特殊的求导方式，不推荐在transformer中使用，这里的代码也不能够在transformer中使用


### 2022/10/19
1. 增加[resample loss](https://github.com/wutong16/DistributionBalancedLoss)
     - 使用前需要执行脚本获取类权重
        ```
        sh ./scrypt/get_class_freq.sh
        ```
2. gcam增加批量处理，每个类结果分别保存的功能
    ```
        python gcam.py --config ./config/NIH_c13.yaml --save_path "./gcam_result"
