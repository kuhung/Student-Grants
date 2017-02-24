### DataCastle 大学生助学金精准资助预测（算法资格赛）第七名解决方案
[网址](http://www.pkbigdata.com/common/cmpt/%E5%A4%A7%E5%AD%A6%E7%94%9F%E5%8A%A9%E5%AD%A6%E9%87%91%E7%B2%BE%E5%87%86%E8%B5%84%E5%8A%A9%E9%A2%84%E6%B5%8B_%E5%8F%82%E8%B5%9B%E4%B8%8E%E7%BB%84%E9%98%9F.html#teamStandard)


#### 成员：
kuhung
Yes,boy!
jacker

#### 代码使用说明：
##### kuhung文件下：
- base_line：基本模型，依次运行3个feature和code，可得榜上0.02867结果。
- input：生成的中间文件。包括特征，top特征索引。
- output：二分类结果与最后提交文件。
- test：测试集文件
- train：训练集文件

- 其余ipynb：包括二分类模型、特征生成模型、特征筛选模型、二分类+多分类合并模型、调参模型。

- top_pro02867.csv: 线上0.02867结果的概率，用于和Yes,boy!的最佳线上概率结果融合。融合模型位于Yes,boy!model3，融合比例为6:4。融合线上结果0.02880。

#### Yes,boy!文件：
- model1：基本模型
- model2：特征生成模型
- model3: 融合模型

															·

**将结果为0.02880的结果导入kuhung文件夹下threshold for bash line模型，与binary_model生成的二分类结果求交，即可得最终线上结果0.02888。**



2016年11月3日-2017年2月20日 
