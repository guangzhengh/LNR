import re

with open('datasets\data_txt\ImageNet_LT_test.txt', 'r+') as f:
    content = f.read()
    f.seek(0)
    f.write(re.sub(r'val/n[^/]+/', 'val/', content))
    f.truncate()