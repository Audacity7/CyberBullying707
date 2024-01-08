# you can run each cell seperately or can run the whole script altogether
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import pandas as pd 
from bs4 import BeautifulSoup
import string
import spacy
import re


# In[2]:


def title_extract_h1(Url):
    # Make a request to the webpage
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:60.0) Gecko/20100101 Firefox/60.0"}
    page = requests.get(URL, headers=headers)
    soup=BeautifulSoup(page.content, 'html.parser')
    #extract title from the article
    title=soup.find('h1', class_=["entry-title", "tdb-title-text", "custom-title"])
    title=title.text.replace('\n'," ")
    return title


# In[3]:


# Read the Excel file
data = pd.read_excel('Input.xlsx')
article_titles = []
# Iterate over each row and extract the article text
for index, row in data.iterrows():
    url_id = row['URL_ID']
    URL = row['URL']
    try:
        title = title_extract_h1(URL)
        article_titles.append(title)
    except Exception as e:
        pass
print (*article_titles, sep="\n")


# In[4]:


article_dict = {}
def text_extract(URL):
    try:
        # Make a request to the webpage
        headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:60.0) Gecko/20100101 Firefox/60.0"}
        page = requests.get(URL, headers=headers)
        soup=BeautifulSoup(page.content, 'html.parser')

        # Check the response status code
        if page.status_code == 200:
            # Webpage exists, continue with scraping
            content=soup.findAll(attrs={'class':'td-post-content'})
            content=content[0].text.replace('\n'," ")
            file_name = f'{url_id}.txt'
            
            # Save the article text to a text file
            with open(file_name, 'w', encoding="utf8") as file:
                file.write(content)
                article_dict.update({url_id:content})
            print(f'Saved article {file_name}')
            pass
        
        elif page.status_code == 404:
            # Webpage does not exist
            print("Page not found!")
        
        else:
            # Handle other response status codes if needed
            print("Unexpected response:", response.status_code)
            print(f'Saved article {file_name}')
        
    except requests.exceptions.RequestException as e:
        # Handle any other request-related exceptions
        print("Error occurred:", str(e))
    return article_dict


# In[5]:


# Read the Excel file
data = pd.read_excel('Input.xlsx')
my_dict = {}
# Iterate over each row and extract the article text
for index, row in data.iterrows():
    url_id = row['URL_ID']
    URL = row['URL']
    text_extract(URL)
    my_dict.update(article_dict)
print('Extraction complete.')


# In[6]:


def translate_content(val):
    content = val.translate(str.maketrans('', '', string.punctuation)) 
    return content


# In[7]:


new_dict = {}

for key, value in my_dict.items():
    # Perform operations on the value
    without_punct = translate_content(value)
    
    # Store the new value in the new dictionary
    new_dict[key] = without_punct

print(new_dict)


# In[8]:


data['Content'] = data['URL_ID'].map(new_dict)
data.head()


# In[9]:


from nltk.tokenize import word_tokenize
token = []
col_values = data["Content"].values
for value in col_values:
    text_tokens = word_tokenize(str(value))
    token.append(text_tokens)
    print(text_tokens[0:50])


# In[10]:


token_len = []
for value in token:
    length = len(value)
    token_len.append(length)
print(token_len)


# In[11]:


data["TokenLength"] = token_len
data.head()


# In[12]:


import nltk
from nltk.corpus import stopwords
without_stpwrds = []
for values in token:
    my_stop_words = stopwords.words('english')
    my_stop_words.append('the')
    no_stop_tokens = [word for word in values if not word in my_stop_words]
    without_stpwrds.append(no_stop_tokens)
    print(no_stop_tokens[0:40])


# In[13]:


without_stpwords_len = []
for value in without_stpwrds:
    length = len(value)
    without_stpwords_len.append(length)
print(without_stpwords_len)


# In[14]:


data["TokenLengthw/oStopwords"] = without_stpwords_len
data.head()


# In[15]:


with open("positive-words.txt","r") as pos:
    poswords = pos.read().split("\n")  
    poswords = poswords[5:]


# In[16]:


pos_score = []
for no_stop_tokens in without_stpwrds:
    pos_count = " ".join ([w for w in no_stop_tokens if w in poswords])
    pos_count=pos_count.split(" ")
    Positive_score=len(pos_count)
    pos_score.append(Positive_score)
print(pos_score)


# In[17]:


data["POSITIVE SCORE"] = pos_score
data.head()


# In[18]:


with open("negative-words.txt","r",encoding = "ISO-8859-1") as neg:
    negwords = neg.read().split("\n")
    
negwords = negwords[36:]


# In[19]:


neg_score = []
for no_stop_tokens in without_stpwrds:
    neg_count = " ".join ([w for w in no_stop_tokens if w in negwords])
    neg_count=neg_count.split(" ")
    Negative_score=len(neg_count)
    neg_score.append(Negative_score)
print(neg_score)


# In[20]:


data["NEGATIVE SCORE"] = neg_score
data.head()


# In[21]:


from textblob import TextBlob
column_values = data['Content'].values

# Compute polarity score for each value in the column
polarity_scores = [TextBlob(str(value)).sentiment.polarity for value in column_values]
subjectivity_scores = [TextBlob(str(value)).sentiment.subjectivity for value in column_values]

polarity_scores_rounded = [round(score, 3) for score in polarity_scores]
subjectivity_scores_rounded = [round(score, 3) for score in subjectivity_scores]

# Print the polarity scores
print(polarity_scores_rounded)
print(subjectivity_scores_rounded)


# In[22]:


data["POLARITY SCORE"] = polarity_scores_rounded
data["SUBJECTIVITY SCORE"] = subjectivity_scores_rounded
data.head()


# In[23]:


#AVG SENTENCE LENGTH
avg_sentence = []
for value in column_values:
    AVG_SENTENCE_LENGTH = len(str(value).replace(' ',''))/len(re.split(r'[?!.]', str(value)))
    avg_sentence.append(AVG_SENTENCE_LENGTH)
    print('Word average =', AVG_SENTENCE_LENGTH)


# In[24]:


data["AVG SENTENCE LENGTH"] = avg_sentence
data.head()


# In[25]:


#complex word count function
def syllable_count(word):
    count = 0
    vowels = "AEIOUYaeiouy"
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)): 
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
            if word.endswith("es"or "ed"):
                count -= 1
    if count == 0:
        count += 1
    return count


# In[26]:


#word count
word_count = []
for value in column_values:
    Word_Count=len(str(value))
    word_count.append(Word_Count)
print(word_count)


# In[27]:


# complex word count
com_word = []
for value in column_values:
    COMPLEX_WORDS=syllable_count(str(value))
    com_word.append(COMPLEX_WORDS)
print(com_word)


# In[28]:


#percentage of complex words
com_percent = []
for value in column_values:
    Word_Count=len(str(value))
    COMPLEX_WORDS=syllable_count(str(value))
    pcw=(COMPLEX_WORDS/Word_Count)*100
    rounded_pcw = round(pcw, 3)
    com_percent.append(rounded_pcw)
print(com_percent)


# In[29]:


# fog index
import textstat
fog_id = []
for value in column_values:
    FOG_INDEX=(textstat.gunning_fog(str(value)))
    fog_id.append(FOG_INDEX)
print(fog_id)


# In[30]:


# average number of words per sentence
avg_word_per_sen = []
for value in column_values:
    AVG_NUMBER_OF_WORDS_PER_SENTENCE = [len(l.split()) for l in re.split(r'[?!.]', str(value)) if l.strip()]
    AVG_NUMBER_OF_WORDS_PER_SENTENCE=sum(AVG_NUMBER_OF_WORDS_PER_SENTENCE)/len(AVG_NUMBER_OF_WORDS_PER_SENTENCE)
    avg_word_per_sen.append(AVG_NUMBER_OF_WORDS_PER_SENTENCE)
print(avg_word_per_sen)


# In[31]:


# syllable per word
syl_per_word = []
for value in column_values:
    word=str(value).replace(' ','')
    syllable_count=0
    for w in word:
          if(w=='a' or w=='e' or w=='i' or w=='o' or w=='y' or w=='u' or w=='A' or w=='E' or w=='I' or w=='O' or w=='U' or w=='Y'):
                syllable_count=syllable_count+1
    
    avg_syl = syllable_count/len(str(value).split())
    avg_syl_rounded = round(avg_syl, 3)
    syl_per_word.append(avg_syl_rounded)
print(syl_per_word)


# In[32]:


import nltk
nltk.download('averaged_perceptron_tagger')


# In[33]:


# personal pronoun function
def PersonalPronounExtractor(text):
    count = 0
    sentences = nltk.sent_tokenize(text)
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        tagged = nltk.pos_tag(words)
        for (word, tag) in tagged:
            if tag == 'PRP': # If the word is a proper noun
                count = count + 1 
        
    return(count)         
                


# Calling the PersonalPronounExtractor function to extract all the proper nouns from the given text. 
personal_pro =[]
for value in column_values:
    Personal_Pronouns=PersonalPronounExtractor(str(value))
    personal_pro.append(Personal_Pronouns)
print(personal_pro)


# In[34]:


# average word length
avg_word_len =[]
for value in column_values:
    Average_Word_Length=len(str(value).replace(' ',''))/len(str(value).split())
    Average_Word_Length_rounded = round(Average_Word_Length, 3)
    avg_word_len.append(Average_Word_Length_rounded)
print(avg_word_len)


# In[35]:


data["PERCENTAGE OF COMPLEX WORDS"] = com_percent
data["FOG INDEX"] = fog_id
data["AVG NUMBER OF WORDS PER SENTENCE"] = avg_word_per_sen
data["COMPLEX WORD COUNT"] = com_word
data["WORD COUNT"] = word_count
data["SYLLABLE PER WORD"] = syl_per_word
data["PERSONAL PRONOUNS"] = personal_pro
data["AVG WORD LENGTH"] = avg_word_len
data.head()


# In[36]:


columns_to_delete = ['Content', 'TokenLength', 'TokenLengthw/oStopwords']

# Drop the specified columns
newData = data.drop(columns_to_delete, axis=1)
newData.head()

# this file contains all the code of all cells from jupyter ipynb file, to avoid any possible errors please run code on jupyter notebook using attached ipynb file.
#!/usr/bin/env python
# coding: utf-8

# In[4]:


import requests
import pandas as pd 
from bs4 import BeautifulSoup
import string
import spacy
import re


# In[5]:


def title_extract_h1(Url):
    # Make a request to the webpage
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:60.0) Gecko/20100101 Firefox/60.0"}
    page = requests.get(URL, headers=headers)
    soup=BeautifulSoup(page.content, 'html.parser')
    #extract title from the article
    title=soup.find('h1', class_=["entry-title", "tdb-title-text", "custom-title"])
    title=title.text.replace('\n'," ")
    return title


# In[6]:


# Read the Excel file
data = pd.read_excel('Input.xlsx')
article_titles = []
# Iterate over each row and extract the article text
for index, row in data.iterrows():
    url_id = row['URL_ID']
    URL = row['URL']
    try:
        title = title_extract_h1(URL)
        article_titles.append(title)
    except Exception as e:
        pass
print (*article_titles, sep="\n")


# In[7]:


article_dict = {}
def text_extract(URL):
    try:
        # Make a request to the webpage
        headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:60.0) Gecko/20100101 Firefox/60.0"}
        page = requests.get(URL, headers=headers)
        soup=BeautifulSoup(page.content, 'html.parser')

        # Check the response status code
        if page.status_code == 200:
            # Webpage exists, continue with scraping
            content=soup.findAll(attrs={'class':'td-post-content'})
            content=content[0].text.replace('\n'," ")
            file_name = f'{url_id}.txt'
            
            # Save the article text to a text file
            with open(file_name, 'w', encoding="utf8") as file:
                file.write(content)
                article_dict.update({url_id:content})
            print(f'Saved article {file_name}')
            pass
        
        elif page.status_code == 404:
            # Webpage does not exist
            print("Page not found!")
        
        else:
            # Handle other response status codes if needed
            print("Unexpected response:", response.status_code)
            print(f'Saved article {file_name}')
        
    except requests.exceptions.RequestException as e:
        # Handle any other request-related exceptions
        print("Error occurred:", str(e))
    return article_dict


# In[8]:


# Read the Excel file
data = pd.read_excel('Input.xlsx')
my_dict = {}
# Iterate over each row and extract the article text
for index, row in data.iterrows():
    url_id = row['URL_ID']
    URL = row['URL']
    text_extract(URL)
    my_dict.update(article_dict)
print('Extraction complete.')


# In[9]:


def translate_content(val):
    content = val.translate(str.maketrans('', '', string.punctuation)) 
    return content


# In[10]:


new_dict = {}

for key, value in my_dict.items():
    # Perform operations on the value
    without_punct = translate_content(value)
    
    # Store the new value in the new dictionary
    new_dict[key] = without_punct

print(new_dict)


# In[11]:


data['Content'] = data['URL_ID'].map(new_dict)
data.head()


# In[12]:


from nltk.tokenize import word_tokenize
token = []
col_values = data["Content"].values
for value in col_values:
    text_tokens = word_tokenize(str(value))
    token.append(text_tokens)
    print(text_tokens[0:50])


# In[13]:


token_len = []
for value in token:
    length = len(value)
    token_len.append(length)
print(token_len)


# In[14]:


data["TokenLength"] = token_len
data.head()


# In[15]:


import nltk
from nltk.corpus import stopwords
without_stpwrds = []
for values in token:
    my_stop_words = stopwords.words('english')
    my_stop_words.append('the')
    no_stop_tokens = [word for word in values if not word in my_stop_words]
    without_stpwrds.append(no_stop_tokens)
    print(no_stop_tokens[0:40])


# In[16]:


without_stpwords_len = []
for value in without_stpwrds:
    length = len(value)
    without_stpwords_len.append(length)
print(without_stpwords_len)


# In[17]:


data["TokenLengthw/oStopwords"] = without_stpwords_len
data.head()


# In[18]:


with open("positive-words.txt","r") as pos:
    poswords = pos.read().split("\n")  
    poswords = poswords[5:]


# In[19]:


pos_score = []
for no_stop_tokens in without_stpwrds:
    pos_count = " ".join ([w for w in no_stop_tokens if w in poswords])
    pos_count=pos_count.split(" ")
    Positive_score=len(pos_count)
    pos_score.append(Positive_score)
print(pos_score)


# In[20]:


data["POSITIVE SCORE"] = pos_score
data.head()


# In[21]:


with open("negative-words.txt","r",encoding = "ISO-8859-1") as neg:
    negwords = neg.read().split("\n")
    
negwords = negwords[36:]


# In[22]:


neg_score = []
for no_stop_tokens in without_stpwrds:
    neg_count = " ".join ([w for w in no_stop_tokens if w in negwords])
    neg_count=neg_count.split(" ")
    Negative_score=len(neg_count)
    neg_score.append(Negative_score)
print(neg_score)


# In[23]:


data["NEGATIVE SCORE"] = neg_score
data.head()


# In[24]:


from textblob import TextBlob
column_values = data['Content'].values

# Compute polarity score for each value in the column
polarity_scores = [TextBlob(str(value)).sentiment.polarity for value in column_values]
subjectivity_scores = [TextBlob(str(value)).sentiment.subjectivity for value in column_values]

polarity_scores_rounded = [round(score, 3) for score in polarity_scores]
subjectivity_scores_rounded = [round(score, 3) for score in subjectivity_scores]

# Print the polarity scores
print(polarity_scores_rounded)
print(subjectivity_scores_rounded)


# In[25]:


data["POLARITY SCORE"] = polarity_scores_rounded
data["SUBJECTIVITY SCORE"] = subjectivity_scores_rounded
data.head()


# In[26]:


#AVG SENTENCE LENGTH
avg_sentence = []
for value in column_values:
    AVG_SENTENCE_LENGTH = len(str(value).replace(' ',''))/len(re.split(r'[?!.]', str(value)))
    avg_sentence.append(AVG_SENTENCE_LENGTH)
    print('Word average =', AVG_SENTENCE_LENGTH)


# In[27]:


data["AVG SENTENCE LENGTH"] = avg_sentence
data.head()


# In[28]:


#complex word count function
def syllable_count(word):
    count = 0
    vowels = "AEIOUYaeiouy"
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)): 
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
            if word.endswith("es"or "ed"):
                count -= 1
    if count == 0:
        count += 1
    return count


# In[29]:


#word count
word_count = []
for value in column_values:
    Word_Count=len(str(value))
    word_count.append(Word_Count)
print(word_count)


# In[30]:


# complex word count
com_word = []
for value in column_values:
    COMPLEX_WORDS=syllable_count(str(value))
    com_word.append(COMPLEX_WORDS)
print(com_word)


# In[31]:


#percentage of complex words
com_percent = []
for value in column_values:
    Word_Count=len(str(value))
    COMPLEX_WORDS=syllable_count(str(value))
    pcw=(COMPLEX_WORDS/Word_Count)*100
    rounded_pcw = round(pcw, 3)
    com_percent.append(rounded_pcw)
print(com_percent)


# In[33]:


# fog index
import textstat
fog_id = []
for value in column_values:
    FOG_INDEX=(textstat.gunning_fog(str(value)))
    fog_id.append(FOG_INDEX)
print(fog_id)


# In[35]:


# average number of words per sentence
avg_word_per_sen = []
for value in column_values:
    AVG_NUMBER_OF_WORDS_PER_SENTENCE = [len(l.split()) for l in re.split(r'[?!.]', str(value)) if l.strip()]
    AVG_NUMBER_OF_WORDS_PER_SENTENCE=sum(AVG_NUMBER_OF_WORDS_PER_SENTENCE)/len(AVG_NUMBER_OF_WORDS_PER_SENTENCE)
    avg_word_per_sen.append(AVG_NUMBER_OF_WORDS_PER_SENTENCE)
print(avg_word_per_sen)


# In[41]:


# syllable per word
syl_per_word = []
for value in column_values:
    word=str(value).replace(' ','')
    syllable_count=0
    for w in word:
          if(w=='a' or w=='e' or w=='i' or w=='o' or w=='y' or w=='u' or w=='A' or w=='E' or w=='I' or w=='O' or w=='U' or w=='Y'):
                syllable_count=syllable_count+1
    
    avg_syl = syllable_count/len(str(value).split())
    avg_syl_rounded = round(avg_syl, 3)
    syl_per_word.append(avg_syl_rounded)
print(syl_per_word)


# In[44]:


import nltk
nltk.download('averaged_perceptron_tagger')


# In[45]:


# personal pronoun function
def PersonalPronounExtractor(text):
    count = 0
    sentences = nltk.sent_tokenize(text)
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        tagged = nltk.pos_tag(words)
        for (word, tag) in tagged:
            if tag == 'PRP': # If the word is a proper noun
                count = count + 1 
        
    return(count)         
                


# Calling the PersonalPronounExtractor function to extract all the proper nouns from the given text. 
personal_pro =[]
for value in column_values:
    Personal_Pronouns=PersonalPronounExtractor(str(value))
    personal_pro.append(Personal_Pronouns)
print(personal_pro)


# In[49]:


# average word length
avg_word_len =[]
for value in column_values:
    Average_Word_Length=len(str(value).replace(' ',''))/len(str(value).split())
    Average_Word_Length_rounded = round(Average_Word_Length, 3)
    avg_word_len.append(Average_Word_Length_rounded)
print(avg_word_len)


# In[50]:


data["PERCENTAGE OF COMPLEX WORDS"] = com_percent
data["FOG INDEX"] = fog_id
data["AVG NUMBER OF WORDS PER SENTENCE"] = avg_word_per_sen
data["COMPLEX WORD COUNT"] = com_word
data["WORD COUNT"] = word_count
data["SYLLABLE PER WORD"] = syl_per_word
data["PERSONAL PRONOUNS"] = personal_pro
data["AVG WORD LENGTH"] = avg_word_len
data.head()


# In[51]:

# to drop unnecessary columns to output file
columns_to_delete = ['Content', 'TokenLength', 'TokenLengthw/oStopwords']

# Drop the specified columns
newData = data.drop(columns_to_delete, axis=1)
newData.head()



# In[52]:


#convert dataframe to xlsx file
data.to_excel('output_Blackcoffer.xlsx', index=False)
print("DataFrame converted to xlsx file.")
