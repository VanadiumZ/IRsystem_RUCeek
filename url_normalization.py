with open('D:\Code\CodeTrainingGSAI\day1\origintext.html', 'r', encoding='utf8') as file:
    html_content = file.read()

from bs4 import BeautifulSoup
soup = BeautifulSoup(html_content, 'html.parser')

# focusing on url normalization

# get all the hrefs
all_links = set()
for anchor in soup.find_all('a'):
    href = anchor.attrs.get("href")
    if (href != "" and href != None):
        all_links.add(href)

# print(all_links)
from url_normalize import url_normalize
base_url = 'http://ai.ruc.edu.cn/'
links = set()
for href in all_links:
    if not href.startswith('http'):
        href = url_normalize(base_url + href)
    # links.add(href)
    print(href)
