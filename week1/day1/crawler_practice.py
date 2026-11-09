import requests
from bs4 import BeautifulSoup

# 1. 发请求
resp = requests.get('https://www.ithome.com/', headers={'User-Agent': 'Mozilla/5.0'})
resp.encoding = 'utf-8'
html = resp.text
print(html[:3000])

# 2. 解析
soup = BeautifulSoup(html, 'lxml')

print("页面长度:", len(html))          # 确认有没有拿到网页
print("h2 数量:", len(soup.find_all('h2')))  # 确认 h2 是不是 0

# 3. 提取
for h2 in soup.find_all('h2'):
    print(h2.get_text(strip=True))
