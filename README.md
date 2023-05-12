# 5.10
- 要考虑dis的影响，可能会直接降低了zi的权重比例。
- dis_range=[1,1.5,2.0],去倒数就是0.5-1.0
- zi的范围是0-1
- 版本质量l的取值范围是1-5

1. 现在的问题是：环境依赖于每次的Fov提供的tile信息，如何减少这种特定环境下的过拟合呢
（好像又没有）

2. 先固定Pmax，求maxQoE；

# 5.11
- 如果超过了计算能力，不done掉，作为惩罚项
- 若Tu和Td大于了Tslot，也只做惩罚，不done掉
- done是正常结束，terminal是发生特殊情况下的终止；done没有nextstate，但terminal有

# 5.12
- rt:1800Mps
- Dmax:60Mbit
- 压缩后25Mbit，未压缩30Mbit

# 5.13
- 结果action始终是[0,1]，网络根本没学到策略