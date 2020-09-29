# 导入依赖包
import re
import os
import requests
from lxml import etree


# 获取网页的url
def get_url(url):
    # 伪造请求头
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.132 Safari/537.36'
    }
    # 发送请求
    res = requests.get(url, headers=headers)
    # 指定编码方式
    res.encoding = 'gbk'
    # 解析网页
    html = etree.HTML(res.text)
    # 对url进行去重
    url_list = set()
    # 提取节点信息
    link = html.xpath('//ul/li[@class="left-list_li"]/a/@href')
    # 将url添加到url列表
    for i in link:
        url_list.add(i)
    link2 = html.xpath('//dl[@class="hot public-box"]/dd/a/@href')
    for j in link2:
        url_list.add(j)
    link3 = html.xpath('//dl[@class="channel public-box"]/dd/a/@href')
    for k in link3:
        url_list.add(k)
    link4 = html.xpath('//ul[@class="column public-box"]/li[@class="column-li"]/a/@href')
    for l in link4:
        url_list.add(l)
    for url in url_list:
        get_img(url)


# 获取图片
def get_img(link):
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.132 Safari/537.36'
    }
    print(link)
    res = requests.get(link, headers=headers)
    res.encoding = 'gbk'
    # 从网页提取需要的信息
    page = re.findall('>共(.*?)页<', res.text)[0]
    res = etree.HTML(res.text)
    img_link = res.xpath('//div/a/img/@src')[0]
    name = res.xpath('//div[@class="content"]/h5/text()')[0]
    # 构造url
    base_url = img_link.split('/')
    url = base_url[:-1]
    url = '/'.join(url)
    for i in range(1, int(page) + 1):
        img_link = url + '/' + str(i) + '.jpg'
        save_img(link, img_link, name, i)


# 保存图片
def save_img(link, img_link, name, i):
    headers = {
        'Referer': link,
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.132 Safari/537.36'
    }
    img = requests.get(img_link, headers=headers)
    # 如果文件夹不存在则新建
    if not os.path.exists('mnsy'):
        os.mkdir('mnsy')
    # 指定图片保存路径和名称
    path = './mnsy/' + name + str(i) + '.jpg'
    print(path)
    # 如果图片已经保存，则直接跳过该图片
    if os.access(path, os.F_OK):
        print("已经存在")
    else:
        try:
            with open(path, 'wb') as f:
                f.write(img.content)
                print('图片存储成功')
        except:
            pass


def main():
    # 要爬取的目标网址
    url = 'https://www.mm131.net/'
    get_url(url)


if __name__ == '__main__':
    main()
