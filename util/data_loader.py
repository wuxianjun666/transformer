from torchtext.legacy.data import Field, BucketIterator
from torchtext.legacy.datasets.translation import Multi30k
#torchtext是一个可以与PyTorch深度结合的文本处理库。它可以方便地对文本进行预处理，如截断补齐、构建词表等
# Field：指定要如何处理某个字段，比如指定分词方法，是否转成小写，起始字符，结束字符，补全字符以及词典等。

class DataLoader:
    source: Field = None
    target: Field = None

    def __init__(self, ext, tokenize_en, tokenize_de, init_token, eos_token):
        self.ext = ext
        self.tokenize_en = tokenize_en
        self.tokenize_de = tokenize_de
        self.init_token = init_token
        self.eos_token = eos_token
        print('dataset initializing start')

    def make_dataset(self):
        if self.ext == ('.de', '.en'):   #tokenize为定义的分词器函数 在每句话的开头加入字符SOS，结尾加入字符EOS，将所有单词转换为小写。
            self.source = Field(tokenize=self.tokenize_de, init_token=self.init_token, eos_token=self.eos_token,
                                lower=True, batch_first=True)
            self.target = Field(tokenize=self.tokenize_en, init_token=self.init_token, eos_token=self.eos_token,
                                lower=True, batch_first=True)

        elif self.ext == ('.en', '.de'):
            self.source = Field(tokenize=self.tokenize_en, init_token=self.init_token, eos_token=self.eos_token,
                                lower=True, batch_first=True)
            self.target = Field(tokenize=self.tokenize_de, init_token=self.init_token, eos_token=self.eos_token,
                                lower=True, batch_first=True)

        train_data, valid_data, test_data = Multi30k.splits(exts=self.ext, fields=(self.source, self.target))  #// splits方法可以同时加载训练集，验证集和测试集，//参数exts指定使用哪种语言作为源语言和目标语言，fileds指定定义好的Field类
        return train_data, valid_data, test_data

    def build_vocab(self, train_data, min_freq):
        self.source.build_vocab(train_data, min_freq=min_freq)
        self.target.build_vocab(train_data, min_freq=min_freq)

    def make_iter(self, train, validate, test, batch_size, device):
        train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train, validate, test),
                                                                              batch_size=batch_size,
                                                                              device=device)
        print('dataset initializing done')
        return train_iterator, valid_iterator, test_iterator
'''
BucketIterator：
相比于标准迭代器，会将类似长度的样本当做一批来处理,
因为在文本处理中经常会需要将每一批样本长度补齐为当前批中最长序列的长度，
因此当样本长度差别较大时，使用BucketIerator可以带来填充效率的提高。
除此之外，我们还可以在Field中通过fix_length参数来对样本进行截断补齐操作。
'''