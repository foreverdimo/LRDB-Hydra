# 基于Hydra的LRDB代码框架
代码框架基于https://github.com/victoresque/pytorch-template/tree/hydra

论文链接https://ieeexplore.ieee.org/document/9055013

## 特点
- 层级式的配置文件
- 完备的日志系统，输出文件夹定制，tensorboard可视化
- 保存性能最好的k个权重而不是全部保存

## 用法
#### 修改配置文件
所有的配置都在conf/文件夹下，请按需修改，注意data下的数据集路径请补充完整

#### 训练
```bash
CUDA_VISIBLE_DEVICES=0 python train.py
```

#### 命令行覆盖配置参数
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py ngpu=4 learning_rate=0.0001
```

#### 修改输出文件夹格式
输出文件夹格式在conf/hydra/run/dir/job_timestamp.yaml内修改，例如想要以学习率命名输出文件夹，可将其内容修改为
```yaml
# @package hydra.run
dir: ./outputs/${status}/${now:%Y-%m-%d}/lr${learning_rate}
```

#### 在PIPAL验证数据集上测试并输出结果

请先按需修改conf/test.yaml，尤其是验证数据集路径和模型权重

```bash
CUDA_VISIBLE_DEVICES=0 python test.py
```

会在相应文件夹内生成output.txt，与readme.txt一起打包后可以按要求在比赛官网提交结果

