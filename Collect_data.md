
# Data Collection
Collect messages from whatsapp


```python
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
import sklearn
import itertools
import re
import string
import csv
```

Crawling data from phone files


```python
df = pd.read_csv('C:\\Users\\DELL\\Desktop\\data_scientist\\data\\data4.txt', sep="\n", header = None, error_bad_lines=False)
df['gender'] = pd.read_csv('C:\\Users\\DELL\\Desktop\\data_scientist\\data\\target.txt', sep="\n", header = None, error_bad_lines=False)

df2 = pd.read_csv('C:\\Users\\DELL\\Desktop\\data_scientist\\data\\eden.txt', sep="\n", header = None, error_bad_lines=False)
df2['gender']=1
df2=df2[1:5000]

df3 = pd.read_csv('C:\\Users\\DELL\\Desktop\\data_scientist\\data\\proj1.txt', sep="\n", header = None, error_bad_lines=False)
df3['gender']=0
df3=df3[1:5000]

frame=[df,df2,df3]
df=pd.concat(frame)
```

    b'Skipping line 8342: expected 1 fields, saw 2\n'
    

# Preproccesing

1. drop null
2. remove invalid messages (with Invalid format)
4. extract message
5. drop duplicates records
6. messages cleaning
7. create and remove stop-words
 


```python
#drop null
df[0]=df[0].dropna()
#remove invalid messages
df=df[df[0].str.contains("<מדיה הושמטה>") == False]
df=df[df[0].str.contains("-") == True]
df[1]=pd.DataFrame(df[0].apply(lambda x: x[x.index('-')+1:]))
df=df[df[1].str.contains(":") == True]
df['message']=pd.DataFrame(df[1].apply(lambda x: x[x.index(':')+1:]))
#drop duplicates records
df.drop_duplicates(subset=None, keep='first', inplace=True)
```


```python
# remove punctuation from data
regex = re.compile('[%s]' % re.escape(string.punctuation))
df['message_clean']= df['message'].apply(lambda x: regex.sub('', x))
```


```python
from collections import Counter
#creat stop-words - the x most frequent words 
result=df.message_clean.apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0)
result=result.sort_values(ascending=False)
stop=result.head(500)
#remove stop-words
df['message_clean'] = df['message_clean'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
```


```python
stop.head(20)
```




                12703.0
    לא           1282.0
    אני          1184.0
    את            666.0
    מה            648.0
    זה            626.0
    לי            550.0
    יש            433.0
    אתה           323.0
    אנונימוס      322.0
    עם            313.0
    חחח           311.0
    אז            291.0
    על            286.0
    גם            285.0
    של            285.0
    כן            276.0
    לך            265.0
    אם            231.0
    אבל           229.0
    dtype: float64




```python
df.head(3)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>gender</th>
      <th>1</th>
      <th>message</th>
      <th>message_clean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>24/04/17, 21:39 - אנונימוס: חחחחח</td>
      <td>0</td>
      <td>אנונימוס: חחחחח</td>
      <td>חחחחח</td>
      <td></td>
    </tr>
    <tr>
      <th>3</th>
      <td>24/04/17, 21:39 - אנונימוס: חברים, התחלנו. בקב...</td>
      <td>0</td>
      <td>אנונימוס: חברים, התחלנו. בקבוצה זו נעסוק בנוש...</td>
      <td>חברים, התחלנו. בקבוצה זו נעסוק בנושאי הליבה ה...</td>
      <td>התחלנו בקבוצה נעסוק בנושאי הליבה הטמונים המממש...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>24/04/17, 21:39 - אנונימוס: מי זאת החמודה לידה</td>
      <td>0</td>
      <td>אנונימוס: מי זאת החמודה לידה</td>
      <td>מי זאת החמודה לידה</td>
      <td>החמודה לידה</td>
    </tr>
  </tbody>
</table>
</div>




```python
import csv
#save to text files
with open('C:\\Users\\DELL\\Desktop\\data_model\\y1.txt', 'w', newline='\n', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file, delimiter='\n')
    writer.writerow(df['message'].values)
with open('C:\\Users\\DELL\\Desktop\\data_model\\y2.txt', 'w', newline='\n', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file, delimiter='\n')
    writer.writerow(df['gender'].values)
```
