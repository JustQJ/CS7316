run the code
```shell
python pagerank.py --method sparse2 #把web_links.csv数据集放在当前文件夹下
```
--method 有三个选择：standard, sparse1, sparse2。
standard使用标准的矩阵乘法迭代，需要巨大的内存，大约600GB，一般无法使用。
sparse1使用课上讲的稀疏矩阵迭代方式，计算速度相对较慢
sparse2使用`scipy.sparse`库的`csr_matrix`存储格式，有高效的稀疏矩阵向量乘法算子，计算速度较快。
测试结果（在64 cpu cores, 256GB内存的设备上测试的结果，迭代30次）
|Method|Memory (MB)|Time (s)|
|---|---|---|
|standard|600*1024+|无法运行测量|
|sparse1|361.3|43.1|
|sparse2|362.2|3.7|

