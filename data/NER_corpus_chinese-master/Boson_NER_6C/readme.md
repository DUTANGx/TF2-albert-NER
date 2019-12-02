## BosonNLP命名实体识别数据

命名实体识别（NER）是指识别文本中具有特定意义的实体，主要包括人名、地名、机构名、专有名词等。命名实体识别是信息提取、问答系统、句法分析、机器翻译等应用领域的重要基础工具，作为结构化信息提取的重要步骤。

在BosonNLP命名实体的标注中，文本采用UTF-8进行编码，每行为一个段落标注，共包括2000段落。所有的实体以如下的格式进行标注：
{{实体类型：实体文本}}

标注的实体类别包括以下6种：

- time: 时间
- location: 地点
- person_name: 人名
- org_name: 组织名
- company_name: 公司名
- product_name: 产品名

例：</p>
此次{{location:中国}}个展，{{person_name:苏珊?菲利普斯}}将与她80多岁高龄的父亲一起合作，哼唱一首古老的{{location:威尔士}}民歌{{product_name:《白蜡林》}}。届时在{{location:画廊大厅}}中将安放6个音箱进行播放，艺术家还特意回到家乡{{location:格拉斯哥}}，同父亲一起在{{org_name:中国音乐学院}}里为作品录制了具有{{location:中国}}元素的音乐片段。

来源： https://bosonnlp.com/resources/BosonNLP_NER_6C.zip
