# skirt_length_design baseline实验描述
天池fashionai比赛地址：https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.100066.0.0.350cd780qWQYjg&raceId=231649
* 将官方提供的`skirt_length_design`的数据拆成训练集和验证集。
* 将所有图片直接resize成224x224
* label直接去看y的位置，忽视m
* 网络结构使用resnet18
* 最后模型收敛的时候，在验证集上的accuracy约90%

# 安装 tensorcv
```bash
$ git clone https://github.com/tworuler/tensorcv.git ~/github
pip install -r requirements.txt
pip install -e .
```
# 启动skirt_length_design baseline实验

## 实验数据准备
* 整理官方的提供的数据。如:
  * 将2次提供的训练集合成在一起放在`~/fashionai/data/train_data`下。
  * 将测试集放在`~/fashionai/data/test_data`下。
* 准备数据列表
  * 筛选出skirt_length相关的数据
  * 将数据拆成训练集和验证集
  ```
  cd ~/fashionai/data/train_data/Annotations
  # label.csv官方提供的第一批训练数据
  cat skirt_length_labels.csv label.csv | grep skirt_length > skirt.csv
  shuf -n 1000 skirt.csv > val.csv
  grep -F -v -f val.csv skirt.csv > train.csv
  ```


## 准备实验配置
* 建立实验目录，并复制baseline实验配置
```bash
mkdir -p ~/fashionai/skirt_length/E01
cd ~/fashionai/skirt_length/E01
cp ~/github/tensorcv/exmaples/fashionai/skirt_length.cfg .
```
* 将skirt_length.cfg中的路径相关配置改成自己的路径。如：
  * 将`PATH_OF_EXPERIMENT_TO_BE_CONFIGURED`改成`~/fashionai/skirt_length/E01`
  * 将`PATH_OF_TRAIN_DATA_FOLDER_BE_CONFIGURED`改成`~/fashionai/data/train_data`
  * 将`PATH_OF_TEST_DATA_FOLDER_BE_CONFIGURED`改成`~/fashionai/data/test_data`
  * 将`PATH_OF_TRAIN_CSV_TO_BE_CONFIGURED`改成`~/fashionai/data/train_data/Annotations/train.csv`
  * 将`PATH_OF_VAL_CSV_TO_BE_CONFIGURED`改成`~/fashionai/data/train_data/Annotations/val.csv`
  * 将`PATH_OF_TEST_CSV_TO_BE_CONFIGURED`改成`~/fashionai/data/test_data/Tests/question.csv`

## 启动实验训练
* 使用下面命令启动实验
```bash
tcv train skirt_length_01.cfg
```

## 生成测试集的answer
* 使用下面命令对测试集inference，会只生成skirt_length相关的答案，在`eval/20000/test_0222.csv`
```bash
tcv predict skirt_length_01.cfg
```

## 使用tensorboard查看实验相关指标
```
tensorboard --logdir . --port 6006
```
