# -*- coding: utf-8 -*-
import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe"
testdata_dir_config = '--tessdata-dir "C:\\Program Files (x86)\\Tesseract-OCR\\tessdata"'
textCode = pytesseract.image_to_string(Image.open(r"F:/test/aaaa.jpg"), lang='eng', config=testdata_dir_config)
print(textCode)
