# GAN-based-speech-enhancement-Keras-
Keras implementation of speech enhancement based on LSGAN
本文仿真内容参考文章：SEGAN:SpeechEnhancementGenerativeAdversarialNetwork
一开始完全按照上文仿真时，计算量太大，我的渣电脑直接死机，后面我对网络输入输出以及结构做了一点简化。

生成器部分有两个输入，inputs是noisy信号，是需要经过神经网络进行增强的信号
另一个输入inputs1是clean信号，该部分信号仅参与loss function的计算

训练阶段noisy和clean信号为1024点一帧，16K采样率。50%重叠。需要准备的数据格式为（帧数，1024）
测试数据没有重叠，其他与训练阶段数据一致。数据可以用matlab生成.mat格式保存。

目前版本写的比较简陋。后续我会用python里面的生成器实时生成不同信噪比的数据再用keras里的fit.generator进行训练。
敬请关系后续更新！！


