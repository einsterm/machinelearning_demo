# -*- encoding:utf-8 -*-

from lxml import etree, html
import sys

reload(sys)
sys.setdefaultencoding('utf-8')
content = open("1.html", "rb").read()
page = html.document_fromstring(content)
text = page.text_content()
print text
