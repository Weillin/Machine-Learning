file_name = 'test.md'
out_name = file_name.replace('md', 'py')

with open(file_name, 'r', encoding='utf-8') as f:
    text = f.read()
    text = text.replace('```python', '')
    text = text.replace('```', '')

with open(out_name, 'w', encoding='utf-8') as f:
    f.write(text)
