### 1.导入依赖包


```python
import re
import os
import time
import random
import requests
from lxml import etree
```

### 2.获取url


```python
def get_url(url):
    # 伪造请求头
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.132 Safari/537.36'
    }
    # 请求网页
    res = requests.get(url, headers=headers)
    #设置休眠时间
    time.sleep(random.randint(1, 3))
    res.encoding = 'gbk'
    html = etree.HTML(res.text)
    page = html.xpath('//dd[@class="page"]/a/@href')[-1]
    page = page.split('.')[0]
    page = page.split('_')[-1]
    link = html.xpath('//dd[@class="page"]/a/@href')[0]
    link = link.split('.')[0]
    link = link[:-1]
    link_list = [url]
    for i in range(2, int(page) + 1):
        u = url + link + str(i) + '.html'
        link_list.append(u)
    for u in link_list:
        parse_url(u)
```

### 3.解析url


```python
def parse_url(u):
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.132 Safari/537.36'
    }
    res = requests.get(u, headers=headers)
    #设置休眠时间
    time.sleep(random.randint(1, 3))
    res.encoding = 'gbk'
    res = etree.HTML(res.text)
    nodes = res.xpath('//dl[@class="list-left public-box"]/dd/a[@target="_blank"]/@href')
    names = res.xpath('//dl[@class="list-left public-box"]/dd/a[@target="_blank"]/img/@alt')
    for name, node in zip(names, nodes):
        path = 'mnfl/' + name + '1.jpg'
        print(path)
        if os.access(path, os.F_OK):
            print("图片已经存在")
            pass
        else:
            print(node)
            get_pic(node)
```

### 4.获取图片


```python
def get_pic(node):
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.132 Safari/537.36'
    }
    # print(link)
    res = requests.get(node, headers=headers)
    #设置休眠时间
    time.sleep(random.randint(1, 3))
    res.encoding = 'gbk'
    page = re.findall('>共(.*?)页<', res.text)[0]
    res = etree.HTML(res.text)
    img_link = res.xpath('//div/a/img/@src')[0]
    name = res.xpath('//div[@class="content"]/h5/text()')[0]
    base_url = img_link.split('/')
    url = base_url[:-1]
    url = '/'.join(url)
    for i in range(1, int(page) + 1):
        img_link = url + '/' + str(i) + '.jpg'
        save_img(node, img_link, name, i)
```

### 5.保存图片


```python
def save_img(link, img_link, name, i):
    headers = {
        'Referer': link,
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.132 Safari/537.36'
    }
    img = requests.get(img_link, headers=headers)
    #设置休眠时间
    time.sleep(random.randint(1, 3))
    if not os.path.exists('mnfl'):
        os.mkdir('mnfl')
    path = 'mnfl/' + name + str(i) + '.jpg'
    print(path)
    if os.access(path, os.F_OK):
        print("图片已经存在")
    else:
        try:
            with open(path, 'wb') as f:
                f.write(img.content)
                print('图片存储成功')
        except:
            print('图片存储失败')
```


```python
def main():
    # 想要哪个分类直接把'qingchun'改为其他类即可
    # url = 'https://www.mm131.net/qingchun/'
    # get_url(url)
    # 如果要爬取所有的类，注释上面两行，运行下面代码
    url_list = ['xinggan', 'qingchun', 'xiaohua', 'chemo', 'qipao', 'mingxing']
    for i in url_list:
        url = 'https://www.mm131.net/' + i + '/'
        get_url(url)
```


```python
if __name__ == '__main__':
    main()
```
