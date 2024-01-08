import urllib.request
from bs4 import BeautifulSoup
 
# here we have to pass url and path
# (where you want to save your text file)
headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:60.0) Gecko/20100101 Firefox/60.0"}
urllib.request.urlretrieve("https://insights.blackcoffer.com/ai-in-healthcare-to-improve-patient-outcomes/",
                           "text_file.txt")
 
file = open("text_file.txt", "r", encoding="utf8")
contents = file.read()
soup = BeautifulSoup(contents, 'html.parser')
 
f = open("test1.txt", "w")
 
# traverse paragraphs from soup
for data in soup.find_all("p"):
    sum = data.get_text()
    f.writelines(sum)
 
f.close()
