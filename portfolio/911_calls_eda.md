---
layout: notebook
title: 911 Calls - Exploratory Data Analysis
skills: Python, Pandas, Seaborn
external_type: Github
external_url: https://github.com/sajal2692/data-science-portfolio/blob/master/911%20Calls%20-%20Exploratory%20Analysis.ipynb
description: Exploratory Data Analysis of the 911 calls dataset hosted on Kaggle. Demonstrates extraction of useful features from different variables.
---
---

## Introduction

For this project we'll analyze the 911 call dataset from [Kaggle](https://www.kaggle.com/mchirico/montcoalert). The data contains the following fields:

* lat : String variable, Latitude
* lng: String variable, Longitude
* desc: String variable, Description of the Emergency Call
* zip: String variable, Zipcode
* title: String variable, Title
* timeStamp: String variable, YYYY-MM-DD HH:MM:SS
* twp: String variable, Township
* addr: String variable, Address
* e: String variable, Dummy variable (always 1)

Let's start with some data analysis and visualisation imports.


```python
import numpy as np
import pandas as pd
```


```python
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

sns.set_style('whitegrid')

plt.rcParams['figure.figsize'] = (6, 4)
```


```python
#Reading the data
df = pd.read_csv('data/911.csv')
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 99492 entries, 0 to 99491
    Data columns (total 9 columns):
    lat          99492 non-null float64
    lng          99492 non-null float64
    desc         99492 non-null object
    zip          86637 non-null float64
    title        99492 non-null object
    timeStamp    99492 non-null object
    twp          99449 non-null object
    addr         98973 non-null object
    e            99492 non-null int64
    dtypes: float64(3), int64(1), object(5)
    memory usage: 6.8+ MB



```python
#Checking the head of the dataframe
df.head()
```




<div>
<table border="1" class="dataframe table-responsive">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lat</th>
      <th>lng</th>
      <th>desc</th>
      <th>zip</th>
      <th>title</th>
      <th>timeStamp</th>
      <th>twp</th>
      <th>addr</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>40.297876</td>
      <td>-75.581294</td>
      <td>REINDEER CT &amp; DEAD END;  NEW HANOVER; Station ...</td>
      <td>19525.0</td>
      <td>EMS: BACK PAINS/INJURY</td>
      <td>2015-12-10 17:40:00</td>
      <td>NEW HANOVER</td>
      <td>REINDEER CT &amp; DEAD END</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>40.258061</td>
      <td>-75.264680</td>
      <td>BRIAR PATH &amp; WHITEMARSH LN;  HATFIELD TOWNSHIP...</td>
      <td>19446.0</td>
      <td>EMS: DIABETIC EMERGENCY</td>
      <td>2015-12-10 17:40:00</td>
      <td>HATFIELD TOWNSHIP</td>
      <td>BRIAR PATH &amp; WHITEMARSH LN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>40.121182</td>
      <td>-75.351975</td>
      <td>HAWS AVE; NORRISTOWN; 2015-12-10 @ 14:39:21-St...</td>
      <td>19401.0</td>
      <td>Fire: GAS-ODOR/LEAK</td>
      <td>2015-12-10 17:40:00</td>
      <td>NORRISTOWN</td>
      <td>HAWS AVE</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>40.116153</td>
      <td>-75.343513</td>
      <td>AIRY ST &amp; SWEDE ST;  NORRISTOWN; Station 308A;...</td>
      <td>19401.0</td>
      <td>EMS: CARDIAC EMERGENCY</td>
      <td>2015-12-10 17:40:01</td>
      <td>NORRISTOWN</td>
      <td>AIRY ST &amp; SWEDE ST</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>40.251492</td>
      <td>-75.603350</td>
      <td>CHERRYWOOD CT &amp; DEAD END;  LOWER POTTSGROVE; S...</td>
      <td>NaN</td>
      <td>EMS: DIZZINESS</td>
      <td>2015-12-10 17:40:01</td>
      <td>LOWER POTTSGROVE</td>
      <td>CHERRYWOOD CT &amp; DEAD END</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## Basic Analysis

Let's check out the top 5 zipcodes for calls.


```python
df['zip'].value_counts().head(5)
```




    19401.0    6979
    19464.0    6643
    19403.0    4854
    19446.0    4748
    19406.0    3174
    Name: zip, dtype: int64



The top townships for the calls were as follows:


```python
df['twp'].value_counts().head(5)
```




    LOWER MERION    8443
    ABINGTON        5977
    NORRISTOWN      5890
    UPPER MERION    5227
    CHELTENHAM      4575
    Name: twp, dtype: int64



For 90k + entries, how many unique call titles did we have? 


```python
df['title'].nunique()
```




    110



## Data Wrangling for Feature Creation

We can extract some generalised features from the columns in our dataset for further analysis. 

In the _title_ column, there's a kind of 'subcategory' or 'reason for call' alloted to each entry (denoted by the text before the colon). 

The timestamp column can be further segregated into Year, Month and Day of Week too. 

Let's start with creating a 'Reason' feature for each call.


```python
df['Reason'] = df['title'].apply(lambda x: x.split(':')[0])
```


```python
df.tail()
```




<div>
<table border="1" class="dataframe table-responsive">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lat</th>
      <th>lng</th>
      <th>desc</th>
      <th>zip</th>
      <th>title</th>
      <th>timeStamp</th>
      <th>twp</th>
      <th>addr</th>
      <th>e</th>
      <th>Reason</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>99487</th>
      <td>40.132869</td>
      <td>-75.333515</td>
      <td>MARKLEY ST &amp; W LOGAN ST; NORRISTOWN; 2016-08-2...</td>
      <td>19401.0</td>
      <td>Traffic: VEHICLE ACCIDENT -</td>
      <td>2016-08-24 11:06:00</td>
      <td>NORRISTOWN</td>
      <td>MARKLEY ST &amp; W LOGAN ST</td>
      <td>1</td>
      <td>Traffic</td>
    </tr>
    <tr>
      <th>99488</th>
      <td>40.006974</td>
      <td>-75.289080</td>
      <td>LANCASTER AVE &amp; RITTENHOUSE PL; LOWER MERION; ...</td>
      <td>19003.0</td>
      <td>Traffic: VEHICLE ACCIDENT -</td>
      <td>2016-08-24 11:07:02</td>
      <td>LOWER MERION</td>
      <td>LANCASTER AVE &amp; RITTENHOUSE PL</td>
      <td>1</td>
      <td>Traffic</td>
    </tr>
    <tr>
      <th>99489</th>
      <td>40.115429</td>
      <td>-75.334679</td>
      <td>CHESTNUT ST &amp; WALNUT ST;  NORRISTOWN; Station ...</td>
      <td>19401.0</td>
      <td>EMS: FALL VICTIM</td>
      <td>2016-08-24 11:12:00</td>
      <td>NORRISTOWN</td>
      <td>CHESTNUT ST &amp; WALNUT ST</td>
      <td>1</td>
      <td>EMS</td>
    </tr>
    <tr>
      <th>99490</th>
      <td>40.186431</td>
      <td>-75.192555</td>
      <td>WELSH RD &amp; WEBSTER LN;  HORSHAM; Station 352; ...</td>
      <td>19002.0</td>
      <td>EMS: NAUSEA/VOMITING</td>
      <td>2016-08-24 11:17:01</td>
      <td>HORSHAM</td>
      <td>WELSH RD &amp; WEBSTER LN</td>
      <td>1</td>
      <td>EMS</td>
    </tr>
    <tr>
      <th>99491</th>
      <td>40.207055</td>
      <td>-75.317952</td>
      <td>MORRIS RD &amp; S BROAD ST; UPPER GWYNEDD; 2016-08...</td>
      <td>19446.0</td>
      <td>Traffic: VEHICLE ACCIDENT -</td>
      <td>2016-08-24 11:17:02</td>
      <td>UPPER GWYNEDD</td>
      <td>MORRIS RD &amp; S BROAD ST</td>
      <td>1</td>
      <td>Traffic</td>
    </tr>
  </tbody>
</table>
</div>



Now, let's find out the most common reason for 911 calls, according to our dataset.


```python
df['Reason'].value_counts()
```




    EMS        48877
    Traffic    35695
    Fire       14920
    Name: Reason, dtype: int64




```python
sns.countplot(df['Reason'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1165ad710>




![png](/public/project-images/911_calls_eda/output_20_1.png)


Let's deal with the time information we have. Checking the datatype of the timestamp column.


```python
type(df['timeStamp'][0])
```




    str



As the timestamps are still string types, it'll make our life easier if we convert it to a python DateTime object, so we can extract the year, month, and day information more intuitively. 


```python
df['timeStamp'] = pd.to_datetime(df['timeStamp'])
```

For a single DateTime object, we can extract information as follows.


```python
time = df['timeStamp'].iloc[0]

print('Hour:',time.hour)
print('Month:',time.month)
print('Day of Week:',time.dayofweek)
```

    Hour: 17
    Month: 12
    Day of Week: 3


Now let's create new features for the above pieces of information.


```python
df['Hour'] = df['timeStamp'].apply(lambda x: x.hour)
df['Month'] = df['timeStamp'].apply(lambda x: x.month)
df['Day of Week'] = df['timeStamp'].apply(lambda x: x.dayofweek)
```


```python
df.head(3)
```




<div>
<table border="1" class="dataframe table-responsive">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lat</th>
      <th>lng</th>
      <th>desc</th>
      <th>zip</th>
      <th>title</th>
      <th>timeStamp</th>
      <th>twp</th>
      <th>addr</th>
      <th>e</th>
      <th>Reason</th>
      <th>Hour</th>
      <th>Month</th>
      <th>Day of Week</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>40.297876</td>
      <td>-75.581294</td>
      <td>REINDEER CT &amp; DEAD END;  NEW HANOVER; Station ...</td>
      <td>19525.0</td>
      <td>EMS: BACK PAINS/INJURY</td>
      <td>2015-12-10 17:40:00</td>
      <td>NEW HANOVER</td>
      <td>REINDEER CT &amp; DEAD END</td>
      <td>1</td>
      <td>EMS</td>
      <td>17</td>
      <td>12</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>40.258061</td>
      <td>-75.264680</td>
      <td>BRIAR PATH &amp; WHITEMARSH LN;  HATFIELD TOWNSHIP...</td>
      <td>19446.0</td>
      <td>EMS: DIABETIC EMERGENCY</td>
      <td>2015-12-10 17:40:00</td>
      <td>HATFIELD TOWNSHIP</td>
      <td>BRIAR PATH &amp; WHITEMARSH LN</td>
      <td>1</td>
      <td>EMS</td>
      <td>17</td>
      <td>12</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>40.121182</td>
      <td>-75.351975</td>
      <td>HAWS AVE; NORRISTOWN; 2015-12-10 @ 14:39:21-St...</td>
      <td>19401.0</td>
      <td>Fire: GAS-ODOR/LEAK</td>
      <td>2015-12-10 17:40:00</td>
      <td>NORRISTOWN</td>
      <td>HAWS AVE</td>
      <td>1</td>
      <td>Fire</td>
      <td>17</td>
      <td>12</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



The Day of the Week is an integer and it might not be instantly clear which number refers to which Day. We can map that information to a Mon-Sun string.


```python
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
```


```python
df['Day of Week'] = df['Day of Week'].map(dmap)

df.tail(3)
```




<div>
<table border="1" class="dataframe table-responsive">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lat</th>
      <th>lng</th>
      <th>desc</th>
      <th>zip</th>
      <th>title</th>
      <th>timeStamp</th>
      <th>twp</th>
      <th>addr</th>
      <th>e</th>
      <th>Reason</th>
      <th>Hour</th>
      <th>Month</th>
      <th>Day of Week</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>99489</th>
      <td>40.115429</td>
      <td>-75.334679</td>
      <td>CHESTNUT ST &amp; WALNUT ST;  NORRISTOWN; Station ...</td>
      <td>19401.0</td>
      <td>EMS: FALL VICTIM</td>
      <td>2016-08-24 11:12:00</td>
      <td>NORRISTOWN</td>
      <td>CHESTNUT ST &amp; WALNUT ST</td>
      <td>1</td>
      <td>EMS</td>
      <td>11</td>
      <td>8</td>
      <td>Wed</td>
    </tr>
    <tr>
      <th>99490</th>
      <td>40.186431</td>
      <td>-75.192555</td>
      <td>WELSH RD &amp; WEBSTER LN;  HORSHAM; Station 352; ...</td>
      <td>19002.0</td>
      <td>EMS: NAUSEA/VOMITING</td>
      <td>2016-08-24 11:17:01</td>
      <td>HORSHAM</td>
      <td>WELSH RD &amp; WEBSTER LN</td>
      <td>1</td>
      <td>EMS</td>
      <td>11</td>
      <td>8</td>
      <td>Wed</td>
    </tr>
    <tr>
      <th>99491</th>
      <td>40.207055</td>
      <td>-75.317952</td>
      <td>MORRIS RD &amp; S BROAD ST; UPPER GWYNEDD; 2016-08...</td>
      <td>19446.0</td>
      <td>Traffic: VEHICLE ACCIDENT -</td>
      <td>2016-08-24 11:17:02</td>
      <td>UPPER GWYNEDD</td>
      <td>MORRIS RD &amp; S BROAD ST</td>
      <td>1</td>
      <td>Traffic</td>
      <td>11</td>
      <td>8</td>
      <td>Wed</td>
    </tr>
  </tbody>
</table>
</div>



Let's combine the newly created features, to check out the most common call reasons based on the day of the week.


```python
sns.countplot(df['Day of Week'],hue=df['Reason'])

plt.legend(bbox_to_anchor=(1.25,1))
```




    <matplotlib.legend.Legend at 0x116d9cbe0>




![png](/public/project-images/911_calls_eda/output_34_1.png)


It makes sense for the number of traffic related 911 calls to be the lowest during the weekends, what's also iteresting is that Emergency Service related calls are also low during the weekend.


```python
sns.countplot(df['Month'],hue=df['Reason'])

plt.legend(bbox_to_anchor=(1.25,1))
```




    <matplotlib.legend.Legend at 0x117dd1c88>




![png](/public/project-images/911_calls_eda/output_36_1.png)


Now, let's check out the relationship between the number of calls and the month.


```python
byMonth = pd.groupby(df,by='Month').count()
```


```python
byMonth['e'].plot.line(y='e')
plt.title('Calls per Month')
plt.ylabel('Number of Calls')
```




    <matplotlib.text.Text at 0x1031d65c0>




![png](/public/project-images/911_calls_eda/output_39_1.png)


Using seaborn, let's fit the number of calls to a month and see if there's any concrete correlation between the two.


```python
byMonth.reset_index(inplace=True)
```


```python
sns.lmplot(x='Month',y='e',data=byMonth)
plt.ylabel('Number of Calls')
```




    <matplotlib.text.Text at 0x109aa7fd0>




![png](/public/project-images/911_calls_eda/output_42_1.png)


So, it does seem that there are fewer emergency calls during the holiday seasons.

Let's extract the date from the timestamp, and see behavior in a little more detail.


```python
df['Date']=df['timeStamp'].apply(lambda x: x.date())
```


```python
df.head(2)
```




<div>
<table border="1" class="dataframe table-responsive">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lat</th>
      <th>lng</th>
      <th>desc</th>
      <th>zip</th>
      <th>title</th>
      <th>timeStamp</th>
      <th>twp</th>
      <th>addr</th>
      <th>e</th>
      <th>Reason</th>
      <th>Hour</th>
      <th>Month</th>
      <th>Day of Week</th>
      <th>Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>40.297876</td>
      <td>-75.581294</td>
      <td>REINDEER CT &amp; DEAD END;  NEW HANOVER; Station ...</td>
      <td>19525.0</td>
      <td>EMS: BACK PAINS/INJURY</td>
      <td>2015-12-10 17:40:00</td>
      <td>NEW HANOVER</td>
      <td>REINDEER CT &amp; DEAD END</td>
      <td>1</td>
      <td>EMS</td>
      <td>17</td>
      <td>12</td>
      <td>Thu</td>
      <td>2015-12-10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>40.258061</td>
      <td>-75.264680</td>
      <td>BRIAR PATH &amp; WHITEMARSH LN;  HATFIELD TOWNSHIP...</td>
      <td>19446.0</td>
      <td>EMS: DIABETIC EMERGENCY</td>
      <td>2015-12-10 17:40:00</td>
      <td>HATFIELD TOWNSHIP</td>
      <td>BRIAR PATH &amp; WHITEMARSH LN</td>
      <td>1</td>
      <td>EMS</td>
      <td>17</td>
      <td>12</td>
      <td>Thu</td>
      <td>2015-12-10</td>
    </tr>
  </tbody>
</table>
</div>



Grouping and plotting the data: 


```python
pd.groupby(df,'Date').count()['e'].plot.line(y='e')

plt.legend().remove()
plt.tight_layout()
```


![png](/public/project-images/911_calls_eda/output_47_0.png)


We can also check out the same plot for each reason separately.


```python
pd.groupby(df[df['Reason']=='Traffic'],'Date').count().plot.line(y='e')
plt.title('Traffic')
plt.legend().remove()
plt.tight_layout()
```


![png](/public/project-images/911_calls_eda/output_49_0.png)



```python
pd.groupby(df[df['Reason']=='Fire'],'Date').count().plot.line(y='e')
plt.title('Fire')
plt.legend().remove()
plt.tight_layout()
```


![png](/public/project-images/911_calls_eda/output_50_0.png)



```python
pd.groupby(df[df['Reason']=='EMS'],'Date').count().plot.line(y='e')
plt.title('EMS')
plt.legend().remove()
plt.tight_layout()
```


![png](/public/project-images/911_calls_eda/output_51_0.png)


Let's create a heatmap for the counts of calls on each hour, during a given day of the week.


```python
day_hour = df.pivot_table(values='lat',index='Day of Week',columns='Hour',aggfunc='count')

day_hour
```




<div>
<table border="1" class="dataframe table-responsive">
  <thead>
    <tr style="text-align: right;">
      <th>Hour</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
      <th>20</th>
      <th>21</th>
      <th>22</th>
      <th>23</th>
    </tr>
    <tr>
      <th>Day of Week</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Fri</th>
      <td>275</td>
      <td>235</td>
      <td>191</td>
      <td>175</td>
      <td>201</td>
      <td>194</td>
      <td>372</td>
      <td>598</td>
      <td>742</td>
      <td>752</td>
      <td>...</td>
      <td>932</td>
      <td>980</td>
      <td>1039</td>
      <td>980</td>
      <td>820</td>
      <td>696</td>
      <td>667</td>
      <td>559</td>
      <td>514</td>
      <td>474</td>
    </tr>
    <tr>
      <th>Mon</th>
      <td>282</td>
      <td>221</td>
      <td>201</td>
      <td>194</td>
      <td>204</td>
      <td>267</td>
      <td>397</td>
      <td>653</td>
      <td>819</td>
      <td>786</td>
      <td>...</td>
      <td>869</td>
      <td>913</td>
      <td>989</td>
      <td>997</td>
      <td>885</td>
      <td>746</td>
      <td>613</td>
      <td>497</td>
      <td>472</td>
      <td>325</td>
    </tr>
    <tr>
      <th>Sat</th>
      <td>375</td>
      <td>301</td>
      <td>263</td>
      <td>260</td>
      <td>224</td>
      <td>231</td>
      <td>257</td>
      <td>391</td>
      <td>459</td>
      <td>640</td>
      <td>...</td>
      <td>789</td>
      <td>796</td>
      <td>848</td>
      <td>757</td>
      <td>778</td>
      <td>696</td>
      <td>628</td>
      <td>572</td>
      <td>506</td>
      <td>467</td>
    </tr>
    <tr>
      <th>Sun</th>
      <td>383</td>
      <td>306</td>
      <td>286</td>
      <td>268</td>
      <td>242</td>
      <td>240</td>
      <td>300</td>
      <td>402</td>
      <td>483</td>
      <td>620</td>
      <td>...</td>
      <td>684</td>
      <td>691</td>
      <td>663</td>
      <td>714</td>
      <td>670</td>
      <td>655</td>
      <td>537</td>
      <td>461</td>
      <td>415</td>
      <td>330</td>
    </tr>
    <tr>
      <th>Thu</th>
      <td>278</td>
      <td>202</td>
      <td>233</td>
      <td>159</td>
      <td>182</td>
      <td>203</td>
      <td>362</td>
      <td>570</td>
      <td>777</td>
      <td>828</td>
      <td>...</td>
      <td>876</td>
      <td>969</td>
      <td>935</td>
      <td>1013</td>
      <td>810</td>
      <td>698</td>
      <td>617</td>
      <td>553</td>
      <td>424</td>
      <td>354</td>
    </tr>
    <tr>
      <th>Tue</th>
      <td>269</td>
      <td>240</td>
      <td>186</td>
      <td>170</td>
      <td>209</td>
      <td>239</td>
      <td>415</td>
      <td>655</td>
      <td>889</td>
      <td>880</td>
      <td>...</td>
      <td>943</td>
      <td>938</td>
      <td>1026</td>
      <td>1019</td>
      <td>905</td>
      <td>731</td>
      <td>647</td>
      <td>571</td>
      <td>462</td>
      <td>274</td>
    </tr>
    <tr>
      <th>Wed</th>
      <td>250</td>
      <td>216</td>
      <td>189</td>
      <td>209</td>
      <td>156</td>
      <td>255</td>
      <td>410</td>
      <td>701</td>
      <td>875</td>
      <td>808</td>
      <td>...</td>
      <td>904</td>
      <td>867</td>
      <td>990</td>
      <td>1037</td>
      <td>894</td>
      <td>686</td>
      <td>668</td>
      <td>575</td>
      <td>490</td>
      <td>335</td>
    </tr>
  </tbody>
</table>
<p>7 rows × 24 columns</p>
</div>



** Now create a HeatMap using this new DataFrame. **


```python
sns.heatmap(day_hour)

plt.tight_layout()
```


![png](/public/project-images/911_calls_eda/output_55_0.png)


We see that most calls take place around the end of office hours on weekdays. We can create a clustermap to pair up similar Hours and Days.


```python
sns.clustermap(day_hour)
```




    <seaborn.matrix.ClusterGrid at 0x11c49f320>




![png](/public/project-images/911_calls_eda/output_57_1.png)


And this concludes the exploratory analysis project.
