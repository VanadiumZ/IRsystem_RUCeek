from urllib.request import urlparse
import re

def get_domain(url):
    parsed_url = urlparse(url)
    return parsed_url.netloc

def val_domain(url, target:list):
    domain = get_domain(url)
    if domain in target:
        return True
    else:
        return False

def val_type(url):
        if not url.endswith((".pdf",".doc",".docx",".xls",".xlsx",".ppt",".pptx",
               ".zip",".rar",".7z",".gz",".tar",".jpg",".jpeg",".png",".gif",".webp",
               ".mp4",".avi",".exe",".dmg")):
            return True
        else:
            return False
    
def val_url(url, target):
    if val_domain(url, target) and val_type(url):
    # if val_type(url):
        return True
    else:
        return False
    
# test sample
# url1 = "https://v.ruc.edu.cn/account/help" # False
# url2 = "http://ai.ruc.edu.cn/research/gsaiacademic/" # False
# url3 = "http://ai.ruc.edu.cn/research/science/20231131002.html" # True
# url4 = "http://ai.ruc.edu.cn/jsky/xsky/jzxx/374a3a81005547bf8654611c1bd6234c.htm" # True
# url5 = "https://gsai.ruc.edu.cn/addons/video/video/play.html?id=6" # True
# target = ["ai.ruc.edu.cn",
#           "gsai.ruc.edu.cn",
#           "www.jiqizhixin.com"]
# print(val_url(url1, target))
# print(val_url(url2, target))
# print(val_url(url3, target))
# print(val_url(url4, target))
# print(val_url(url5, target))
# in the following stage, rewrite this file into a class module