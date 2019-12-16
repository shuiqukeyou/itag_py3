# -*- coding=utf8 -*-
# 自定义层文件
from keras import backend as K
from keras.engine.topology import Layer, InputSpec


class Masked(Layer):
    def __init__(self, **kwargs):
        super(Masked, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(Masked, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, mask=None):
        output = x
        if mask is not None:
            # remove padding values
            # floatx()：返回当前默认的浮点类型（字符串）
            # K.cast(x,type)：将张量x转换为指定的type类型
            m = K.cast(mask, K.floatx())
            # K.expand_dims(m, -1)：在m上添加一个尺寸的维度
            output = x * K.expand_dims(m, -1)
        return output

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        return output_shape


class AttentionLayer(Layer):

    def __init__(self, units, return_alphas=False, **kwargs):
        # 官方文档要求调用
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units  # 256
        self.input_dim_en = 0
        self.input_dim_de = 0
        self.input_en_times = 0
        self.return_alphas = return_alphas

    def build(self, input_shape):
        # input_shape:[(None, 100, 256), (None, 5, 256)]
        self.input_dim_en = input_shape[0][-1]  # 编码器的宽度,256
        self.input_en_times = input_shape[0][-2]  # 编码器的条目数,100
        self.input_dim_de = input_shape[1][-1]  # 解码器的宽度,256


        # 编码器隐藏值权重(256*256)
        self.w_en = self.add_weight(name='w_en', shape=(self.input_dim_en, self.units),
                                    initializer='glorot_uniform', trainable=True)
        # 解码器隐藏值权重(256*256)
        self.w_de = self.add_weight(name='w_de', shape=(self.input_dim_de, self.units),
                                    initializer='glorot_uniform', trainable=True)
        # u(256*1)
        self.nu = self.add_weight(name='nu', shape=(self.units, 1),
                                  initializer='glorot_uniform', trainable=True)
        # 官方文档要求调用
        super(AttentionLayer, self).build(input_shape)


    def call(self, inputs, mask=None, **kwargs):
        # 读取编码隐藏序列、解码隐藏序列
        en_seq = inputs[0]  # ? * 100 * 256
        print(1,en_seq.shape)
        de_seq = inputs[1]  # ? * 5 * 256
        print(2,de_seq.shape)
        input_de_times = K.shape(de_seq)[-2]  # 5


        # compute alphas
        # 编码序列
        print(3, K.reshape(en_seq, (-1, self.input_dim_en)).shape)
        print(3, self.w_en.shape)
        att_en = K.dot(K.reshape(en_seq, (-1, self.input_dim_en)), self.w_en)  # ?*100*256
        print(4,att_en.shape)
        att_en = K.reshape(att_en, shape=(-1, self.input_en_times * self.units))  # ?*25600
        print(5,att_en.shape)
        # 将编码器序列复制为解码器输入的tag数
        att_en = K.repeat(att_en, input_de_times)  # ?*5 * 25600
        att_en = K.reshape(att_en, shape=(-1, self.input_en_times * input_de_times, self.units))  # ? * 500 * 256


        # 解码序列
        att_de = K.dot(K.reshape(de_seq, (-1, self.input_dim_de)), self.w_de)  # ? * 5 * 256
        att_de = K.reshape(att_de, shape=(-1, input_de_times, self.units))  # ? * 5 * 256
        # 复制为编码器的条目数(repeat_elements(x, rep, axis))
        att_de = K.repeat_elements(att_de, self.input_en_times, 1)  # ? * 100 * 256

        co_m = att_en + att_de  # 1*100*256
        co_m = K.reshape(co_m, (-1, self.units))  # ? * 100*256

        # uij
        mu = K.dot(K.tanh(co_m), self.nu)  # ? * 500 * 1
        mu = K.reshape(mu, shape=(-1, input_de_times, self.input_en_times))  # ? * 5 * 100
        # 注意力权重
        alphas = K.softmax(mu)  # ? * 5 * 100
        # Indicator Function 权重
        p_gen = K.sigmoid(mu)  # ? * 5 *100

        # 调整编码序列
        en_seq = K.reshape(en_seq, shape=(-1, self.input_en_times * self.input_dim_en))  # ? * 25600
        en_seq = K.repeat(en_seq, input_de_times)  # 5 * 25600
        en_seq = K.reshape(en_seq, shape=(-1, input_de_times, self.input_en_times, self.input_dim_en))  # ? * 5 * 100 * 256

        # 沿第二维度进行加和，获得输出（s'）
        #  K.expand_dims(alphas, -1): 1 * 5 * 100 * 256
        # 1 * 5 * 256
        sum_en = K.sum(en_seq * K.expand_dims(alphas, -1), 2)

        # output = K.concatenate([de_seq, sum_en], -1)
        # 输出(论文将s和s'直接进行了加和)
        output = de_seq + sum_en  # 1 * 5 * 256
        print(output.shape)
        if self.return_alphas:
            alphas = K.reshape(alphas, shape=(-1, input_de_times, self.input_en_times))  # 1 * 5 * 100
            p_gen = K.reshape(p_gen, shape=(-1, input_de_times, self.input_en_times))  # 1 * 5 * 100
            return [output] + [alphas] + [p_gen]
        else:
            return output

        # en_seq = inputs[0]
        # de_seq = inputs[1]
        # input_de_times = K.shape(de_seq)[-2]
        #
        # et = K.tanh(K.dot(en_seq, self.w_en) + K.dot(de_seq, self.w_de))  # 中间值
        # at = K.softmax(K.squeeze(K.dot(et, self.u), axis=-1))  # 注意力权重
        #
        # # 如果传进来的层的有过mask处理，mask变量会记录之前mask处理的参数
        # if mask is not None:
        #     at *= K.cast(mask, K.floatx())
        #
        # atx = K.expand_dims(at, axis=-1)  # 将权重矩阵转换为向量
        # ot = atx * en_seq  # 计算权重加成
        # output = K.sum(ot, axis=1)  # 计算最终输出
        # return output, at  # 返回最终输出和注意力权重

    # 声明支持mask后必须有这个函数
    def compute_mask(self, inputs, mask=None):
        return None

    # 供框架调用，用于推断输出的形状
    def compute_output_shape(self, input_shape):
        # output_shape = (input_shape[1][0], input_shape[1][1], input_shape[0][-1] + input_shape[1][-1])
        output_shape = (input_shape[1][0], input_shape[1][1], input_shape[0][-1])
        if self.return_alphas:
            alpha_shape = [(input_shape[1][0], input_shape[1][1], input_shape[0][1])]
            pgen_shape = [(input_shape[1][0], input_shape[1][1], input_shape[0][1])]
            return [output_shape] + alpha_shape + pgen_shape
        else:
            return output_shape



