import requests
import pandas as pd 
from bs4 import BeautifulSoup
import string
import spacy
import re

url = "https://insights.blackcoffer.com/ai-in-healthcare-to-improve-patient-outcomes/"
headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:60.0) Gecko/20100101 Firefox/60.0"}
page = requests.get(url, headers=headers)
soup=BeautifulSoup(page.content, 'html.parser')

title=soup.find('h3',class_="entry-title")
title=title.text.replace('\n'," ")
title

content=soup.findAll(attrs={'class':'td-post-content'})
content=content[0].text.replace('\n'," ")
content