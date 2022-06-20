import spacy

#用来对文本进行断词、短句、词干化、标注词性、命名实体识别、名词短语提取、基于词向量计算词间相似度等处理。
class Tokenizer:

    def __init__(self):   # 安装语言包(en)
        self.spacy_de = spacy.load('de_core_news_sm')
        self.spacy_en = spacy.load('en_core_web_sm')

    def tokenize_de(self, text):
        """
        Tokenizes German text from a string into a list of strings
        """
        return [tok.text for tok in self.spacy_de.tokenizer(text)]

    def tokenize_en(self, text):
        """
        Tokenizes English text from a string into a list of strings
        """
        return [tok.text for tok in self.spacy_en.tokenizer(text)]
