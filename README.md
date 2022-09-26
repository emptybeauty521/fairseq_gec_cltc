# CLTC2022 fairseq_gec语法纠错
### 依赖  
```
python >= 3.6  
pyTorch >= 1.0.0  
pip install -e ./ (安装项目中的fairseq)
pyhanlp
```
### 生成伪纠错数据  
```bash noise.sh```  
### 预处理为二进制数据  
```bash preprocess.sh```  
### 预训练  
```bash pretrain.sh```  
### 生成微调或精调数据的对齐信息（需要安装fast_align和Moses 4.0）  
```bash align.sh```  
### 微调  
```bash train.sh```  
### 精调  
```bash train_sec.sh```  
### 测试  
```
# 使用多个模型进行预测时模型路径用英文冒号分隔
CUDA_VISIBLE_DEVICES=1 python predict.py --data ./data/cltc/test/cged2021.src \
--output ./data/cltc/test \
--model_path ./model/ft_lang8_all_cged_all_cltc7_7/checkpoint1.pt:./model/ft_lang8_all_cged_all_cltc4_9/checkpoint5.pt:./model/ft_lang8_all_cged_all_cltc4_10/checkpoint5.pt \
--batch_size 128 \
--beam_size 4 \
--round 3
```
