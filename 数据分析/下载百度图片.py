import requests
import re
import os
import math

word = "美女"
num = 100

n = 1
for nu in range(math.ceil(num/60)):
    url = 'https://image.baidu.com/search/acjson?tn=resultjson_com&logid=12092513820975934687&ipn=rj&ct=201326592&is=&fp=result&queryWord=' + word + '&cl=2&lm=-1&ie=utf-8&oe=utf-8&adpicid=&st=-1&z=&ic=0&hd=&latest=&copyright=&word=' + word + '&s=&se=&tab=&width=&height=&face=0&istype=2&qc=&nc=1&fr=&expermode=&force=&pn=' + str(
        nu * 60) + '&rn=60&gsm=10e&1606893497459='
    headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8,zh-TW;q=0.7,en-US;q=0.6,zh-HK;q=0.5',
        'Cache-Control': 'max-age=0',
        'Connection': 'keep-alive',
        'Host': 'image.baidu.com',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'same-origin',
        'Sec-Fetch-User': '?1',
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.66 Safari/537.36',
    }
    res = requests.get(url, headers=headers)
    res.encoding = 'utf-8'
    lik = re.findall(r"https://(.*?).jpg", res.text)
    links = []
    for i in lik:
        link = "https://" + i + ".jpg"
        if link not in links:
            links.append(link)


    for img_link in links:
        headers = {
            ':authority': 'ss1.bdstatic.com',
            ':method': 'GET',
            ':path': '/70cFvXSh_Q1YnxGkpoWK1HF6hhy/it/u=3238956355,1713135268&fm=26&gp=0.jpg',
            ':scheme': 'https',
            'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
            'accept-encoding': 'gzip, deflate, br',
            'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8,zh-TW;q=0.7,en-US;q=0.6,zh-HK;q=0.5',
            'cache-control': 'max-age=0',
            'if-modified-since': 'Thu, 01 Jan 1970 00:00:00 GMT',
            'if-none-match': 'bff1343f4022a637c83e2078037406af',
            'sec-fetch-dest': 'document',
            'sec-fetch-mode': 'navigate',
            'sec-fetch-site': 'none',
            'sec-fetch-user': '?1',
            'upgrade-insecure-requests': '1',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.66 Safari/537.36'
        }
        res = requests.get(img_link, headers)
        if not os.path.exists(word):
            os.mkdir(word)
        path = './' + word + '/' + word + str(n) + '.jpg'
        if n > num:
            break
        n += 1
        print(path)
        if os.access(path, os.F_OK):
            print("已经存在")
        else:
            try:
                with open(path, 'wb') as f:
                    f.write(res.content)
                    print('图片存储成功')
            except:
                print('图片存储失败')
