import json

file_name = 'test.ipynb'
out_name = file_name.replace('ipynb', 'py')

with open(file_name, encoding='utf8') as f:
    text = json.load(f)

with open(out_name, 'w', encoding='utf8') as f:
    for item in text['cells']:
        f.writelines([i.rstrip() + '\n' for i in item['source']])
