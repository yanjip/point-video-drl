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

# 5.14
- 可以强调一下算法的可容错性和弹性能力，因为即使违背了Tu+Td的约束条件，
凭借用户的视频缓冲区内容不为空，依旧不影响用户体验。
- 那需要将Bt看做环境吗，我觉得需要。
- 现在有两种选择，违背约束直接丢弃，或者让他继续发送。

# 5.15
- 计算出来的q值全是nan ：
解决办法：①改激活函数为softmax（没用）
②直接原因是因为next state设置为了nan，改成全0

- 又遇到问题：刚开始测试的时候，就一直action为3
原因：奖励函数没设置对

- 把softmax改回relu又可以学习到策略了

- 关于Bt，如果在某个时隙传输失败，我设定丢弃该tile，因为有Bt缓冲区的存在
仍然不会影响到用户体验。

- 现在仿真不需要把bt作为状态变量！！

- 又出现问题：总是选择质量版本最低那个action要么为4，要么9

# 5.17日
- 改了QoE的计算方法
- 每1k个step之后总是先奖励值会下降：
解决办法--删掉evaluate函数即可！为什么？我也不知道

# 5.19日
- 今天写了baseline，用贪婪算法实现（在evaluate里面）
先预分配所有tile最低的质量，然后依次分配高质量，直到达到资源上限结束
版本的进一步提高。
- 发现一个现象：给tile分配的压缩数量越少，QoE却提高了（在baseline是这样）
- 原因应该是这样：参数设置的不对，导致计算能力不够，通信资源过剩，选择压缩版本反而占用了时隙的时间。

# 5.22日
- 下层策略返回Q给上层，然后可以作为环境给上层agent，然后奖励可以
引导agent不仅max所有QoE，并且保证公平性。

# 5.23日
- 后期可以加上buffer的上限（比如联邦学习提到的3s）

# 6.4日
- 先完成基本的波束赋形，state只加sinr，后期可以考虑加上所有用户累积的QoE

# 6.5日
- 完成了DDPG的整个框架，但模型还有很大问题：
①噪声的处理
②动作的归一化处理（改过了，不知道对不对）（期间还遇到点小麻烦，比如照搬的代码是在pendum那个
环境运行的，他的action只有一个值，所以后面取了个[0,0])
③训练结果显示，有个BS的W为全是0！应该是那个干扰项的原因（）
④W用的实数表示，可能用复数更好
⑤W可以归一化，然后直接乘以发射功率即可

# 6.6日
- 删去了done，DDPG更新的算法删掉就行了
- norm可能为0，所以会输出nan

# 6.7日
- 初始化的W一直保持不变；
- 改Ounoise的非常重要
- 跑出来reward确实在升高，但是算出来的结果sinr并没有很大，说明策略有问题
- APy越多，sinr反而训练出来越大了，应该是干扰更多了
- 测试时发现，每次生成的W都不一样，最后得到的reward居然一样！
而且beamformer都是0.707！！！
- Ounoise中max和min改成一样训练效果明显差得很多；max设的太大（比如0.8）明显效果也不好；
目前最好的效果是0.4-0.1
- 有问题：由于reward是速率的总和，导致有个UE的sinr却很低！

# 6.8日
- 解决UE3太小的问题时：改成var之后，根本学不到策略。
- 可以这样：3个UE分为3个档次，根据累积的数据改变三个UE的顺序。
- test时表现并不好，应该是因为train的时候H没有变动。

# 6.9日
- state加上了信道的值，开始reward还出现复数，原来是计算奖励的时候错误了
- 不收敛不收敛，再改成reward不除以10试试
- 不对，因为每次信道数据不一样，得到的reward如果仍然是信道容量值的话，每次
reward都会因为信道状态不一样而改变，所以难以收敛
- 改成复数之后，actor的激活函数应该要改成tanh比较合适
- 仿真了很多次总结：不用fixW很差，timestamp很小很差

- 下一个目标：放弃信道自适应，考虑公平性的分配。

# 6.10日
- 四篇论文全部没有改变信道数据进行仿真
- 把state改成W和H仍然不行，还是之前的原因，因为信道不一样无法收敛。
- 那就这样了，信道不改变，如何加入用户公平性呢
- 别急，信道改成np.random.rayleigh时，算法上升后又下降才收敛

# 6.11日
- minsigma改成0.1最后训练效果比较稳定。
- 下一步模拟更加贴合的信道数据。

# 6.12日
- 闹了个大乌龙，之前程序会先打印tile信息，一直没管，原来是自动运行了tile那个文件，
还是刚才发现每次运行程序时生成的随机数居然是一样的，这几个主文件都没有seed，原来在
tile那个文件里设置了！

# 6.13日：
- 搞了一上午加下午三点，代码给写崩了；原本用的是服务QoE最小的用户，好像还行；
后面又改成了满足sinr约束，这种情况下有时候会难以学习到策略，所以加上了个warmup，
然后然后就改崩了，现在回退到前天的版本了。
- 之前中途改成了瑞利信道好像效果还不错，可是现在重现不了了！！

- 整理一下思路：我的目标是保证至少两个用户的传输性能比较好（测试显示K=2个用户很差）

