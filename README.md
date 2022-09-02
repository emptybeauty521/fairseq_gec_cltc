# CLTC2022 fairseq_gec语法纠错
1.生成伪纠错数据  
```bash noise.sh```  
2.预处理为二进制数据  
```bash preprocess.sh```  
3.预训练  
```bash pretrain.sh```  
4.生成微调或精调数据的对齐信息（需要安装fast_align和Moses 4.0）  
```bash align.sh```  
5.微调  
```bash train.sh```  
6.精调  
```bash train_sec.sh```  
7.测试  
```python predict.py```
