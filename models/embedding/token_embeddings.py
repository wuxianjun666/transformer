from torch import nn


class TokenEmbedding(nn.Embedding):
    """
    Token Embedding using torch.nn
    they will dense representation of word using weighted matrix
    """

    def __init__(self, vocab_size, d_model):
        """
        一个保存了固定字典和大小的简单查找表。这个模块常用来保存词嵌入和用下标检索它们。模块的输入是一个下标的列表，输出是对应的词嵌入。
        这是一个矩阵类，里面初始化了一个随机矩阵，矩阵的长是字典的大小，宽是用来表示字典中每个元素的属性向量，向量的维度根据你想要表示的元素的复杂度而定。
        输入下标0，输出就是embeds矩阵中第0行。
        class for token embedding that included positional information
        嵌入向量中的值是服从标准正态分布的。
        :param vocab_size: size of vocabulary 典的大小尺寸，比如总共出现5000个词，那就输入5000。此时index为（0-4999）
        :param d_model: dimensions of model  #嵌入向量的维度，即用多少维来表示一个符号。
        """
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)