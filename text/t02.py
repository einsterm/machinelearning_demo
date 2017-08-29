# -*- encoding:utf-8 -*-
import sys
import os
import jieba

reload(sys)
sys.setdefaultencoding('utf-8')

a = jieba.cut("关联文书仅以本网收录的裁判文书为依据，"
              "如果对此结果有疑义，请联系作出裁判文书的人民法院", cut_all=True)
print "000".join(a)
