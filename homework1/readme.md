代码运行命令：
```shell
python lfm_recommendation.py \
    --K 30 \
    --lr 0.001 \
    --l2_lamda1 0.1 \
    --l2_lamda2 0.1 \
    --batch_size 16 \
    --action 0 \
    --optim adam \
    --num_epochs 20 
```
改变action参数的值，使用不同的优化手段。
```
0: with bias and regularization
1: without bias and with regularization
2: without bias and regularization
```