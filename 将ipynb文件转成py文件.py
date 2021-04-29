import json

with open('test.ipynb', encoding='utf8') as f:
    text = json.load(f)

with open('test.py', 'w', encoding='utf8') as f:
    for item in text['cells']:
        f.writelines([i.rstrip() + '\n' for i in item['source']])
