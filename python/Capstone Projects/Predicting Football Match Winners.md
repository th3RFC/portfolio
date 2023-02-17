# Predict Football Match Winners in English Premier League (EPL) 🏆

## Table of Contents:
* [Introduction](#1)
* [Data Collection & Cleaning](#2)
    * [Data Download](#2.1)
    * [Scrape Data for Single Team & Single Season](#2.2)
    * [Scrape Data for Multiple Teams & Multiple Years](#2.3)
    * [Concatenate Data Frames](#2.4)
    * [Display Web-Scraped Data Frame](#2.5)
    * [Data Cleaning](#2.6)
    * [Remove Unnecessary Variables](#2.8)
    * [Create New Variables](#2.9)
* [Machine Learning](#3)
    * [Random Forest](#3.1)
    * [Why Random Forest?](#3.2)
    * [Training Algorithm](#3.3)
    * [Prediction & Accuracy](#3.4)
    * [Improving the Model](#3.5)
    * [Predictions Function](#3.6)
    * [Matching Results](#3.7)
* [Conclusion](#4)
    * [Further Development Suggestions](#4.1)

## Introduction ⚽ <a class="anchor" id="1"></a>

Welcome to this machine learning project with **Python**!

In this project, we will be analysing match data on **English Premier League (EPL)** matches to ultimately try and build a model that will predict football match results.

This project comprises two main sections:

- Web-scraping football match data
- Building machine learning model to predict match results

In the first section, we will be scraping web data on match statics from the [FB Ref](https://fbref.com/en/comps/9/Premier-League-Stats) website. This is an easy-to-use source for football stats including player, team, and league stats.

The second section involves building a machine learning model to try and predict the outcomes of football matches. The model that we will be employing is the **Random Forest** Classifier. After building an initial model, we will assess the predictive accuracy prior to changing the model's hyper-parameters to try and optimize its predictive power.

![Football%20Pitch%20Tactics.PNG](attachment:Football%20Pitch%20Tactics.PNG)

## Data Collection & Cleaning ⚽ <a class="anchor" id="2"></a>

We will be using data on English Premier League football matches. To get the data, we will need to scrape the match results from a website.

To begin, we will download the data using *Python*'s **requests**. This will return *html* text data that we must then parse using the **BeautifulSoup** library, enabling us to extract the relevant statistics tables. Finally, we will load everything into a **pandas** data frame in order to clean and prepare the data for building our machine learning model.

The data we are using will be scraped from the [FB Ref](https://fbref.com/en/comps/9/Premier-League-Stats) website.

### Data Download ⚽ <a class="anchor" id="2.1"></a>


```python
import requests
```


```python
standings_url = "https://fbref.com/en/comps/9/Premier-League-Stats"
```

We now need to use the **.get( )** method to make a request to the server and download *html* as a text file from the webpage.


```python
data = requests.get(standings_url)
```

Using the command **data.text** would return us a very long string of *html*. We will not perform this action, however, as the text is barely legible.

On the website, each squad (team) has a link to its own page, as highlighted below in **red**. On these pages, there is a lot of information about games that has been collected which we will be using to train out machine learning model.

![Capture.PNG](attachment:Capture.PNG)

### Scrape Data for Single Team & Single Season ⚽ <a class="anchor" id="2.2"></a>

Before scraping match data for multiple teams and multiple years, we will run through the actions for a single team and a single season, step-by-step.

On each team's page, there is a match log, and we want to be able to extract this data. In order to extract this data, we need the URLs for each team's page containing the match log. To do this, we will need to parse our html using the **BeautifulSoup** library.

After importing the **BeautifulSoup** library, we will:

1. Create a BeautifulSoup object and initialise this by feeding it the *html* data that we downloaded from the first webpage
2. Use CSS selector to give the object a table to select
3. Select the anchor tags with the links that we want from the table
4. Retrieve **href** property of each link using a list comprehension
5. Filter links to get only squad/ team links


```python
from bs4 import BeautifulSoup
```


```python
# (1) - Create and initialise the object
soup = BeautifulSoup(data.text)

# (2) - Use .select() to get first table with class stats_table
standings_table = soup.select('table.stats_table')[0]

# (3) - Find all <a> tags
links = standings_table.find_all('a')

# (4) - Get href for each link
links = [l.get("href") for l in links]

# (5) - Filter links
links = [l for l in links if '/squads/' in l]
```

The above code will return the links without the domain. To turn our links into full URLs (or *absolute links*), we need to attach the domain name onto the front of each using a *format string*.


```python
team_urls = [f"https://fbref.com{l}" for l in links]
```

We will work with the first team's URL, i.e. the team currently leading in the Premier League.


```python
data = requests.get(team_urls[0])
```

In the team's webpage, there is a table called **Scores & Fixtures** that contains information such as the date of the matches, the goals each team scored, the results, etc. 

In order to retrieve the data from this table, we will turn it into a **pandas** data frame using the **.html( )** method, as follows.


```python
import pandas as pd

# Match scans for a specific string inside the table; this will return a list - take the first element
matches = pd.read_html(data.text, match="Scores & Fixtures")[0]
```

From this page, we also want to extract data on shooting statistics by following the **Shooting** tab link to another webpage.

![Capture2.PNG](attachment:Capture2.PNG)

Let us find the URL of this Shooting page. After that, the process will be similar to what we have done before.


```python
soup = BeautifulSoup(data.text)
links = soup.find_all('a')
links = [l.get("href") for l in links]
links = [l for l in links if l and 'all_comps/shooting/' in l]
```


```python
data = requests.get(f"https://fbref.com{links[0]}")
```


```python
shooting = pd.read_html(data.text, match="Shooting")[0]
```


```python
# Display head of shooting data
shooting.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="10" halign="left">For Manchester City</th>
      <th>...</th>
      <th colspan="4" halign="left">Standard</th>
      <th colspan="5" halign="left">Expected</th>
      <th>Unnamed: 25_level_0</th>
    </tr>
    <tr>
      <th></th>
      <th>Date</th>
      <th>Time</th>
      <th>Comp</th>
      <th>Round</th>
      <th>Day</th>
      <th>Venue</th>
      <th>Result</th>
      <th>GF</th>
      <th>GA</th>
      <th>Opponent</th>
      <th>...</th>
      <th>Dist</th>
      <th>FK</th>
      <th>PK</th>
      <th>PKatt</th>
      <th>xG</th>
      <th>npxG</th>
      <th>npxG/Sh</th>
      <th>G-xG</th>
      <th>np:G-xG</th>
      <th>Match Report</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2021-08-07</td>
      <td>17:15</td>
      <td>Community Shield</td>
      <td>FA Community Shield</td>
      <td>Sat</td>
      <td>Neutral</td>
      <td>L</td>
      <td>0</td>
      <td>1</td>
      <td>Leicester City</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Match Report</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2021-08-15</td>
      <td>16:30</td>
      <td>Premier League</td>
      <td>Matchweek 1</td>
      <td>Sun</td>
      <td>Away</td>
      <td>L</td>
      <td>0</td>
      <td>1</td>
      <td>Tottenham</td>
      <td>...</td>
      <td>16.9</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>1.9</td>
      <td>1.9</td>
      <td>0.11</td>
      <td>-1.9</td>
      <td>-1.9</td>
      <td>Match Report</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2021-08-21</td>
      <td>15:00</td>
      <td>Premier League</td>
      <td>Matchweek 2</td>
      <td>Sat</td>
      <td>Home</td>
      <td>W</td>
      <td>5</td>
      <td>0</td>
      <td>Norwich City</td>
      <td>...</td>
      <td>17.3</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>2.7</td>
      <td>2.7</td>
      <td>0.17</td>
      <td>1.3</td>
      <td>1.3</td>
      <td>Match Report</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2021-08-28</td>
      <td>12:30</td>
      <td>Premier League</td>
      <td>Matchweek 3</td>
      <td>Sat</td>
      <td>Home</td>
      <td>W</td>
      <td>5</td>
      <td>0</td>
      <td>Arsenal</td>
      <td>...</td>
      <td>14.3</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>3.8</td>
      <td>3.8</td>
      <td>0.15</td>
      <td>1.2</td>
      <td>1.2</td>
      <td>Match Report</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021-09-11</td>
      <td>15:00</td>
      <td>Premier League</td>
      <td>Matchweek 4</td>
      <td>Sat</td>
      <td>Away</td>
      <td>W</td>
      <td>1</td>
      <td>0</td>
      <td>Leicester City</td>
      <td>...</td>
      <td>14.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>2.9</td>
      <td>2.9</td>
      <td>0.12</td>
      <td>-1.9</td>
      <td>-1.9</td>
      <td>Match Report</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 26 columns</p>
</div>



If you see above, we appear to have a multi-level index. This will cause problems if we want to index based on - for example - **Round** or **GF**, so we need to remove this index level.


```python
# Drop top index level
shooting.columns = shooting.columns.droplevel()
```

Finally, we must merge the two data frames together using the **.merge( )** method. We only want to merge the following columns, from the Shooting data frame:

- **Date**: Match date
- **Sh** : Shots
- **SoT** : Shots-on-target
- **Dist** : Average distance travelled by a shot
- **FK** : Free-kicks
- **PK** : Penalty kicks
- **PKatt** : Penalty kicks attempted


```python
team_data = matches.merge(shooting[["Date", "Sh", "SoT", "Dist", "FK", "PK", "PKatt"]], on="Date")
```


```python
team_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Time</th>
      <th>Comp</th>
      <th>Round</th>
      <th>Day</th>
      <th>Venue</th>
      <th>Result</th>
      <th>GF</th>
      <th>GA</th>
      <th>Opponent</th>
      <th>...</th>
      <th>Formation</th>
      <th>Referee</th>
      <th>Match Report</th>
      <th>Notes</th>
      <th>Sh</th>
      <th>SoT</th>
      <th>Dist</th>
      <th>FK</th>
      <th>PK</th>
      <th>PKatt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2021-08-07</td>
      <td>17:15</td>
      <td>Community Shield</td>
      <td>FA Community Shield</td>
      <td>Sat</td>
      <td>Neutral</td>
      <td>L</td>
      <td>0</td>
      <td>1</td>
      <td>Leicester City</td>
      <td>...</td>
      <td>4-3-3</td>
      <td>Paul Tierney</td>
      <td>Match Report</td>
      <td>NaN</td>
      <td>12</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2021-08-15</td>
      <td>16:30</td>
      <td>Premier League</td>
      <td>Matchweek 1</td>
      <td>Sun</td>
      <td>Away</td>
      <td>L</td>
      <td>0</td>
      <td>1</td>
      <td>Tottenham</td>
      <td>...</td>
      <td>4-3-3</td>
      <td>Anthony Taylor</td>
      <td>Match Report</td>
      <td>NaN</td>
      <td>18</td>
      <td>4</td>
      <td>16.9</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2021-08-21</td>
      <td>15:00</td>
      <td>Premier League</td>
      <td>Matchweek 2</td>
      <td>Sat</td>
      <td>Home</td>
      <td>W</td>
      <td>5</td>
      <td>0</td>
      <td>Norwich City</td>
      <td>...</td>
      <td>4-3-3</td>
      <td>Graham Scott</td>
      <td>Match Report</td>
      <td>NaN</td>
      <td>16</td>
      <td>4</td>
      <td>17.3</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2021-08-28</td>
      <td>12:30</td>
      <td>Premier League</td>
      <td>Matchweek 3</td>
      <td>Sat</td>
      <td>Home</td>
      <td>W</td>
      <td>5</td>
      <td>0</td>
      <td>Arsenal</td>
      <td>...</td>
      <td>4-3-3</td>
      <td>Martin Atkinson</td>
      <td>Match Report</td>
      <td>NaN</td>
      <td>25</td>
      <td>10</td>
      <td>14.3</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021-09-11</td>
      <td>15:00</td>
      <td>Premier League</td>
      <td>Matchweek 4</td>
      <td>Sat</td>
      <td>Away</td>
      <td>W</td>
      <td>1</td>
      <td>0</td>
      <td>Leicester City</td>
      <td>...</td>
      <td>4-3-3</td>
      <td>Paul Tierney</td>
      <td>Match Report</td>
      <td>NaN</td>
      <td>25</td>
      <td>8</td>
      <td>14.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>



So, what we have done so far is to scrape the standings, and then downloaded match and shooting statistics for a single team before combining this information into a single data frame. 

Next, we need to scale this method up and scrape data for multiple teams for multiple years.

### Scrape Data for Multiple Teams & Multiple Years ⚽ <a class="anchor" id="2.3"></a>

To scrape match data for multiple teams from multiple years, we will need to create a **for loop**.

First, we will create one list object containing the years we wish to pull data from, and also an empty list to store all of our data frames. Each data frame will contain the match logs for one team for one season.


```python
years = list(range(2022, 2020, -1))
all_matches = []

standings_url = "https://fbref.com/en/comps/9/Premier-League-Stats"
```

Now we will write a *for loop*. The below code may seem intimidating but it is predominantly what we have already done with a single team and a single season:

1. Loop through each year in the years list
2. Retrive URLs for each team's webpage
3. Loop through each team's webpage
4. Extract the 'Scores & Fixtures' table
5. Retrieve URLs for Shooting webpage
6. Extract Shooting table

Additional stages are

7. Wrap merge in try-block as some data is not always available and may cause merging issues
8. Filter on Premier League matches
9. Add additional columns to each data frame, to identify each data frame team and year
10. Add data frame to data frame list
11. Sleep for 1 second - this is important as some websites may block you if you make too many requests in a short period



```python
import time

# (1) - Loop through the years
for year in years:
    
    # (2) - Retrieve the links for each team page
    data = requests.get(standings_url)
    soup = BeautifulSoup(data.text)
    standings_table = soup.select('table.stats_table')[0]

    links = [l.get("href") for l in standings_table.find_all('a')]
    links = [l for l in links if '/squads/' in l]
    team_urls = [f"https://fbref.com{l}" for l in links]
    
    # Retrieve the links for previous season
    previous_season = soup.select("a.prev")[0].get("href")
    standings_url = f"https://fbref.com{previous_season}"
    
    # (3) - Loop through each team url
    for team_url in team_urls:
        
        # Extract team name from url
        team_name = team_url.split("/")[-1].replace("-Stats", "").replace("-", " ")
        
        # Retrieve team URL
        data = requests.get(team_url)
        
        # (4) - Extract Scores and Fixtures table
        matches = pd.read_html(data.text, match="Scores & Fixtures")[0]
        
        # (5) - Retrieve URL for Shooting page
        soup = BeautifulSoup(data.text)
        links = [l.get("href") for l in soup.find_all('a')]
        links = [l for l in links if l and 'all_comps/shooting/' in l]
        data = requests.get(f"https://fbref.com{links[0]}")
        
        # (6) - Extract Shooting table 
        shooting = pd.read_html(data.text, match="Shooting")[0]
        shooting.columns = shooting.columns.droplevel()
        
        # (7) - Wrap merge in a try statement
        try:
            team_data = matches.merge(shooting[["Date", "Sh", "SoT", "Dist", "FK", "PK", "PKatt"]], on="Date")
        except ValueError:
            continue
            
        # (8) - Filter for Premier League matches
        team_data = team_data[team_data["Comp"] == "Premier League"]
        
        # (9) - Add additional columns
        team_data["Season"] = year
        team_data["Team"] = team_name
        
        # (10) - Add data frame to all_matches list
        all_matches.append(team_data)
        
        # (11) - Sleep for 1 second
        time.sleep(1)
```


```python
# Check how many data frames we have
len(all_matches)
```




    39



### Concatenate Data Frames ⚽ <a class="anchor" id="2.4"></a>

We have ended up with 39 different data frames that we need to concatenate into one large one.


```python
match_df = pd.concat(all_matches)
```

As a bit of house-keeping, we will also cast each column name to lower-case.


```python
# Lower-case all of the columns
match_df.columns = [c.lower() for c in match_df.columns]
```

### Display Web-Scraped Data Frame ⚽ <a class="anchor" id="2.5"></a>

We finally have the data that we are going to use, stored as a data frame with 1389 observations and 27 variables.


```python
match_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>time</th>
      <th>comp</th>
      <th>round</th>
      <th>day</th>
      <th>venue</th>
      <th>result</th>
      <th>gf</th>
      <th>ga</th>
      <th>opponent</th>
      <th>...</th>
      <th>match report</th>
      <th>notes</th>
      <th>sh</th>
      <th>sot</th>
      <th>dist</th>
      <th>fk</th>
      <th>pk</th>
      <th>pkatt</th>
      <th>season</th>
      <th>team</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2021-08-15</td>
      <td>16:30</td>
      <td>Premier League</td>
      <td>Matchweek 1</td>
      <td>Sun</td>
      <td>Away</td>
      <td>L</td>
      <td>0</td>
      <td>1</td>
      <td>Tottenham</td>
      <td>...</td>
      <td>Match Report</td>
      <td>NaN</td>
      <td>18.0</td>
      <td>4.0</td>
      <td>16.9</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2022</td>
      <td>Manchester City</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2021-08-21</td>
      <td>15:00</td>
      <td>Premier League</td>
      <td>Matchweek 2</td>
      <td>Sat</td>
      <td>Home</td>
      <td>W</td>
      <td>5</td>
      <td>0</td>
      <td>Norwich City</td>
      <td>...</td>
      <td>Match Report</td>
      <td>NaN</td>
      <td>16.0</td>
      <td>4.0</td>
      <td>17.3</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2022</td>
      <td>Manchester City</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2021-08-28</td>
      <td>12:30</td>
      <td>Premier League</td>
      <td>Matchweek 3</td>
      <td>Sat</td>
      <td>Home</td>
      <td>W</td>
      <td>5</td>
      <td>0</td>
      <td>Arsenal</td>
      <td>...</td>
      <td>Match Report</td>
      <td>NaN</td>
      <td>25.0</td>
      <td>10.0</td>
      <td>14.3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2022</td>
      <td>Manchester City</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021-09-11</td>
      <td>15:00</td>
      <td>Premier League</td>
      <td>Matchweek 4</td>
      <td>Sat</td>
      <td>Away</td>
      <td>W</td>
      <td>1</td>
      <td>0</td>
      <td>Leicester City</td>
      <td>...</td>
      <td>Match Report</td>
      <td>NaN</td>
      <td>25.0</td>
      <td>8.0</td>
      <td>14.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2022</td>
      <td>Manchester City</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2021-09-18</td>
      <td>15:00</td>
      <td>Premier League</td>
      <td>Matchweek 5</td>
      <td>Sat</td>
      <td>Home</td>
      <td>D</td>
      <td>0</td>
      <td>0</td>
      <td>Southampton</td>
      <td>...</td>
      <td>Match Report</td>
      <td>NaN</td>
      <td>16.0</td>
      <td>1.0</td>
      <td>15.7</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2022</td>
      <td>Manchester City</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>38</th>
      <td>2021-05-02</td>
      <td>19:15</td>
      <td>Premier League</td>
      <td>Matchweek 34</td>
      <td>Sun</td>
      <td>Away</td>
      <td>L</td>
      <td>0</td>
      <td>4</td>
      <td>Tottenham</td>
      <td>...</td>
      <td>Match Report</td>
      <td>NaN</td>
      <td>8.0</td>
      <td>1.0</td>
      <td>17.4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2021</td>
      <td>Sheffield United</td>
    </tr>
    <tr>
      <th>39</th>
      <td>2021-05-08</td>
      <td>15:00</td>
      <td>Premier League</td>
      <td>Matchweek 35</td>
      <td>Sat</td>
      <td>Home</td>
      <td>L</td>
      <td>0</td>
      <td>2</td>
      <td>Crystal Palace</td>
      <td>...</td>
      <td>Match Report</td>
      <td>NaN</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>11.4</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2021</td>
      <td>Sheffield United</td>
    </tr>
    <tr>
      <th>40</th>
      <td>2021-05-16</td>
      <td>19:00</td>
      <td>Premier League</td>
      <td>Matchweek 36</td>
      <td>Sun</td>
      <td>Away</td>
      <td>W</td>
      <td>1</td>
      <td>0</td>
      <td>Everton</td>
      <td>...</td>
      <td>Match Report</td>
      <td>NaN</td>
      <td>10.0</td>
      <td>3.0</td>
      <td>17.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2021</td>
      <td>Sheffield United</td>
    </tr>
    <tr>
      <th>41</th>
      <td>2021-05-19</td>
      <td>18:00</td>
      <td>Premier League</td>
      <td>Matchweek 37</td>
      <td>Wed</td>
      <td>Away</td>
      <td>L</td>
      <td>0</td>
      <td>1</td>
      <td>Newcastle Utd</td>
      <td>...</td>
      <td>Match Report</td>
      <td>NaN</td>
      <td>11.0</td>
      <td>1.0</td>
      <td>16.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2021</td>
      <td>Sheffield United</td>
    </tr>
    <tr>
      <th>42</th>
      <td>2021-05-23</td>
      <td>16:00</td>
      <td>Premier League</td>
      <td>Matchweek 38</td>
      <td>Sun</td>
      <td>Home</td>
      <td>W</td>
      <td>1</td>
      <td>0</td>
      <td>Burnley</td>
      <td>...</td>
      <td>Match Report</td>
      <td>NaN</td>
      <td>12.0</td>
      <td>3.0</td>
      <td>17.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2021</td>
      <td>Sheffield United</td>
    </tr>
  </tbody>
</table>
<p>1389 rows × 27 columns</p>
</div>



We want to save this to file, so we work on it without having to make unnecessary requests to the host website and risk getting blocked!


```python
# Write to CSV file
match_df.to_csv("matches.csv")
```

If you are following along with the project but are struggling to get the web-scraping part to work, or may have gotten blocked by the web-server, **don't panic**! You can download the data that will be stored in my GitHub folder for this project. This will enable you to proceed with the next parts of the project.

### Data Cleaning ⚽ <a class="anchor" id="2.6"></a>

The next stage in this project is to clean the data. The code in this next section can be run independently of the web-scraping code above. It only assumes that you have downloaded the match.csv file from my GitHub folder.

To begin, we will read the data into our notebook as a **pandas** dataframe.

While we have already imported the *pandas* library in the web-scraping part of the project, we will re-import the module so that this code will work independently of code previously written.


```python
import pandas as pd
```


```python
matches = pd.read_csv("matches.csv", index_col=0)
```


```python
matches.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>time</th>
      <th>comp</th>
      <th>round</th>
      <th>day</th>
      <th>venue</th>
      <th>result</th>
      <th>gf</th>
      <th>ga</th>
      <th>opponent</th>
      <th>...</th>
      <th>match report</th>
      <th>notes</th>
      <th>sh</th>
      <th>sot</th>
      <th>dist</th>
      <th>fk</th>
      <th>pk</th>
      <th>pkatt</th>
      <th>season</th>
      <th>team</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2021-08-15</td>
      <td>16:30</td>
      <td>Premier League</td>
      <td>Matchweek 1</td>
      <td>Sun</td>
      <td>Away</td>
      <td>L</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>Tottenham</td>
      <td>...</td>
      <td>Match Report</td>
      <td>NaN</td>
      <td>18.0</td>
      <td>4.0</td>
      <td>16.9</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2022</td>
      <td>Manchester City</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2021-08-21</td>
      <td>15:00</td>
      <td>Premier League</td>
      <td>Matchweek 2</td>
      <td>Sat</td>
      <td>Home</td>
      <td>W</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>Norwich City</td>
      <td>...</td>
      <td>Match Report</td>
      <td>NaN</td>
      <td>16.0</td>
      <td>4.0</td>
      <td>17.3</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2022</td>
      <td>Manchester City</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2021-08-28</td>
      <td>12:30</td>
      <td>Premier League</td>
      <td>Matchweek 3</td>
      <td>Sat</td>
      <td>Home</td>
      <td>W</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>Arsenal</td>
      <td>...</td>
      <td>Match Report</td>
      <td>NaN</td>
      <td>25.0</td>
      <td>10.0</td>
      <td>14.3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2022</td>
      <td>Manchester City</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021-09-11</td>
      <td>15:00</td>
      <td>Premier League</td>
      <td>Matchweek 4</td>
      <td>Sat</td>
      <td>Away</td>
      <td>W</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>Leicester City</td>
      <td>...</td>
      <td>Match Report</td>
      <td>NaN</td>
      <td>25.0</td>
      <td>8.0</td>
      <td>14.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2022</td>
      <td>Manchester City</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2021-09-18</td>
      <td>15:00</td>
      <td>Premier League</td>
      <td>Matchweek 5</td>
      <td>Sat</td>
      <td>Home</td>
      <td>D</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Southampton</td>
      <td>...</td>
      <td>Match Report</td>
      <td>NaN</td>
      <td>16.0</td>
      <td>1.0</td>
      <td>15.7</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2022</td>
      <td>Manchester City</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>



We can use the **.shape( )** method to find the dimensions of our data frame. This will provide us with information on the number of observations and the number of variables per observation.


```python
matches.shape
```




    (1389, 27)



Thus, we have **1389** observations and **27** variables in our data frame, whereby each observation represents a single match. The variables include statistics on which team won or lost, goals for each team, shooting statistics and so on.

However, in the web-scraping part of the course, we looped over **2 seasons** of EPL matches, with **20 squads** per season, with each team playing **38 matches** per season. Do our numbers add up?


```python
# 2 seasons * 20 squads * 38 matches
1389 - (2 * 20 * 38)
```




    -131



It seems that our data frame is short by 131 observations (matches). Why?

We can use the **.value_counts( )** method to begin investigating this.


```python
# Number of matches per team in EPL
matches["team"].value_counts().to_frame()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>team</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Southampton</th>
      <td>72</td>
    </tr>
    <tr>
      <th>Brighton and Hove Albion</th>
      <td>72</td>
    </tr>
    <tr>
      <th>Manchester United</th>
      <td>72</td>
    </tr>
    <tr>
      <th>West Ham United</th>
      <td>72</td>
    </tr>
    <tr>
      <th>Newcastle United</th>
      <td>72</td>
    </tr>
    <tr>
      <th>Burnley</th>
      <td>71</td>
    </tr>
    <tr>
      <th>Leeds United</th>
      <td>71</td>
    </tr>
    <tr>
      <th>Crystal Palace</th>
      <td>71</td>
    </tr>
    <tr>
      <th>Manchester City</th>
      <td>71</td>
    </tr>
    <tr>
      <th>Wolverhampton Wanderers</th>
      <td>71</td>
    </tr>
    <tr>
      <th>Tottenham Hotspur</th>
      <td>71</td>
    </tr>
    <tr>
      <th>Arsenal</th>
      <td>71</td>
    </tr>
    <tr>
      <th>Leicester City</th>
      <td>70</td>
    </tr>
    <tr>
      <th>Chelsea</th>
      <td>70</td>
    </tr>
    <tr>
      <th>Aston Villa</th>
      <td>70</td>
    </tr>
    <tr>
      <th>Everton</th>
      <td>70</td>
    </tr>
    <tr>
      <th>Liverpool</th>
      <td>38</td>
    </tr>
    <tr>
      <th>Fulham</th>
      <td>38</td>
    </tr>
    <tr>
      <th>West Bromwich Albion</th>
      <td>38</td>
    </tr>
    <tr>
      <th>Sheffield United</th>
      <td>38</td>
    </tr>
    <tr>
      <th>Brentford</th>
      <td>34</td>
    </tr>
    <tr>
      <th>Watford</th>
      <td>33</td>
    </tr>
    <tr>
      <th>Norwich City</th>
      <td>33</td>
    </tr>
  </tbody>
</table>
</div>



We can see that most teams have played approximately 70-72 matches in the EPL. Given that we are part way through a season, this is understandable. 

We can also see 7 teams with 38 or fewer matches played. Each season in the EPL, 3 teams are relegated and move to the **Championship** league, while 3 teams end up being promoted from the Championship to the EPL. This would explain why 6 teams have fewer games played ... but we have 7. Why?

**Liverpool** having only 38 matches played looks a little bit suspect, as Liverpool is a strong team that tends to do well. It can be independently verified that Liverpool has not been relegated/ promoted in the last 2 EPL seasons. This would suggest that not all the matches for Liverpool have been pulled through. To investigate this further, we will filter on Liverpool.


```python
matches[matches["team"] == "Liverpool"].sort_values("date")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>time</th>
      <th>comp</th>
      <th>round</th>
      <th>day</th>
      <th>venue</th>
      <th>result</th>
      <th>gf</th>
      <th>ga</th>
      <th>opponent</th>
      <th>...</th>
      <th>match report</th>
      <th>notes</th>
      <th>sh</th>
      <th>sot</th>
      <th>dist</th>
      <th>fk</th>
      <th>pk</th>
      <th>pkatt</th>
      <th>season</th>
      <th>team</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2020-09-12</td>
      <td>17:30</td>
      <td>Premier League</td>
      <td>Matchweek 1</td>
      <td>Sat</td>
      <td>Home</td>
      <td>W</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>Leeds United</td>
      <td>...</td>
      <td>Match Report</td>
      <td>NaN</td>
      <td>20.0</td>
      <td>4.0</td>
      <td>17.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2021</td>
      <td>Liverpool</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-09-20</td>
      <td>16:30</td>
      <td>Premier League</td>
      <td>Matchweek 2</td>
      <td>Sun</td>
      <td>Away</td>
      <td>W</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>Chelsea</td>
      <td>...</td>
      <td>Match Report</td>
      <td>NaN</td>
      <td>17.0</td>
      <td>5.0</td>
      <td>17.7</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2021</td>
      <td>Liverpool</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-09-28</td>
      <td>20:00</td>
      <td>Premier League</td>
      <td>Matchweek 3</td>
      <td>Mon</td>
      <td>Home</td>
      <td>W</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>Arsenal</td>
      <td>...</td>
      <td>Match Report</td>
      <td>NaN</td>
      <td>21.0</td>
      <td>9.0</td>
      <td>16.8</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2021</td>
      <td>Liverpool</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2020-10-04</td>
      <td>19:15</td>
      <td>Premier League</td>
      <td>Matchweek 4</td>
      <td>Sun</td>
      <td>Away</td>
      <td>L</td>
      <td>2.0</td>
      <td>7.0</td>
      <td>Aston Villa</td>
      <td>...</td>
      <td>Match Report</td>
      <td>NaN</td>
      <td>14.0</td>
      <td>8.0</td>
      <td>15.8</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2021</td>
      <td>Liverpool</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2020-10-17</td>
      <td>12:30</td>
      <td>Premier League</td>
      <td>Matchweek 5</td>
      <td>Sat</td>
      <td>Away</td>
      <td>D</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>Everton</td>
      <td>...</td>
      <td>Match Report</td>
      <td>NaN</td>
      <td>22.0</td>
      <td>8.0</td>
      <td>15.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2021</td>
      <td>Liverpool</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2020-10-24</td>
      <td>20:00</td>
      <td>Premier League</td>
      <td>Matchweek 6</td>
      <td>Sat</td>
      <td>Home</td>
      <td>W</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>Sheffield Utd</td>
      <td>...</td>
      <td>Match Report</td>
      <td>NaN</td>
      <td>17.0</td>
      <td>5.0</td>
      <td>18.2</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2021</td>
      <td>Liverpool</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2020-10-31</td>
      <td>17:30</td>
      <td>Premier League</td>
      <td>Matchweek 7</td>
      <td>Sat</td>
      <td>Home</td>
      <td>W</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>West Ham</td>
      <td>...</td>
      <td>Match Report</td>
      <td>NaN</td>
      <td>8.0</td>
      <td>2.0</td>
      <td>18.6</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2021</td>
      <td>Liverpool</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2020-11-08</td>
      <td>16:30</td>
      <td>Premier League</td>
      <td>Matchweek 8</td>
      <td>Sun</td>
      <td>Away</td>
      <td>D</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Manchester City</td>
      <td>...</td>
      <td>Match Report</td>
      <td>NaN</td>
      <td>9.0</td>
      <td>2.0</td>
      <td>21.5</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2021</td>
      <td>Liverpool</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2020-11-22</td>
      <td>19:15</td>
      <td>Premier League</td>
      <td>Matchweek 9</td>
      <td>Sun</td>
      <td>Home</td>
      <td>W</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>Leicester City</td>
      <td>...</td>
      <td>Match Report</td>
      <td>NaN</td>
      <td>24.0</td>
      <td>12.0</td>
      <td>11.9</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2021</td>
      <td>Liverpool</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2020-11-28</td>
      <td>12:30</td>
      <td>Premier League</td>
      <td>Matchweek 10</td>
      <td>Sat</td>
      <td>Away</td>
      <td>D</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Brighton</td>
      <td>...</td>
      <td>Match Report</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>2.0</td>
      <td>20.9</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2021</td>
      <td>Liverpool</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2020-12-06</td>
      <td>19:15</td>
      <td>Premier League</td>
      <td>Matchweek 11</td>
      <td>Sun</td>
      <td>Home</td>
      <td>W</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>Wolves</td>
      <td>...</td>
      <td>Match Report</td>
      <td>NaN</td>
      <td>11.0</td>
      <td>6.0</td>
      <td>16.6</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2021</td>
      <td>Liverpool</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2020-12-13</td>
      <td>16:30</td>
      <td>Premier League</td>
      <td>Matchweek 12</td>
      <td>Sun</td>
      <td>Away</td>
      <td>D</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Fulham</td>
      <td>...</td>
      <td>Match Report</td>
      <td>NaN</td>
      <td>11.0</td>
      <td>5.0</td>
      <td>20.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2021</td>
      <td>Liverpool</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2020-12-16</td>
      <td>20:00</td>
      <td>Premier League</td>
      <td>Matchweek 13</td>
      <td>Wed</td>
      <td>Home</td>
      <td>W</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>Tottenham</td>
      <td>...</td>
      <td>Match Report</td>
      <td>NaN</td>
      <td>17.0</td>
      <td>11.0</td>
      <td>15.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2021</td>
      <td>Liverpool</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2020-12-19</td>
      <td>12:30</td>
      <td>Premier League</td>
      <td>Matchweek 14</td>
      <td>Sat</td>
      <td>Away</td>
      <td>W</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>Crystal Palace</td>
      <td>...</td>
      <td>Match Report</td>
      <td>NaN</td>
      <td>14.0</td>
      <td>7.0</td>
      <td>13.2</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2021</td>
      <td>Liverpool</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2020-12-27</td>
      <td>16:30</td>
      <td>Premier League</td>
      <td>Matchweek 15</td>
      <td>Sun</td>
      <td>Home</td>
      <td>D</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>West Brom</td>
      <td>...</td>
      <td>Match Report</td>
      <td>NaN</td>
      <td>17.0</td>
      <td>2.0</td>
      <td>17.8</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2021</td>
      <td>Liverpool</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2020-12-30</td>
      <td>20:00</td>
      <td>Premier League</td>
      <td>Matchweek 16</td>
      <td>Wed</td>
      <td>Away</td>
      <td>D</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Newcastle Utd</td>
      <td>...</td>
      <td>Match Report</td>
      <td>NaN</td>
      <td>11.0</td>
      <td>4.0</td>
      <td>16.7</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2021</td>
      <td>Liverpool</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2021-01-04</td>
      <td>20:00</td>
      <td>Premier League</td>
      <td>Matchweek 17</td>
      <td>Mon</td>
      <td>Away</td>
      <td>L</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>Southampton</td>
      <td>...</td>
      <td>Match Report</td>
      <td>NaN</td>
      <td>17.0</td>
      <td>1.0</td>
      <td>14.3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2021</td>
      <td>Liverpool</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2021-01-17</td>
      <td>16:30</td>
      <td>Premier League</td>
      <td>Matchweek 19</td>
      <td>Sun</td>
      <td>Home</td>
      <td>D</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Manchester Utd</td>
      <td>...</td>
      <td>Match Report</td>
      <td>NaN</td>
      <td>17.0</td>
      <td>3.0</td>
      <td>17.1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2021</td>
      <td>Liverpool</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2021-01-21</td>
      <td>20:00</td>
      <td>Premier League</td>
      <td>Matchweek 18</td>
      <td>Thu</td>
      <td>Home</td>
      <td>L</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>Burnley</td>
      <td>...</td>
      <td>Match Report</td>
      <td>NaN</td>
      <td>27.0</td>
      <td>6.0</td>
      <td>17.3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2021</td>
      <td>Liverpool</td>
    </tr>
    <tr>
      <th>30</th>
      <td>2021-01-28</td>
      <td>20:00</td>
      <td>Premier League</td>
      <td>Matchweek 20</td>
      <td>Thu</td>
      <td>Away</td>
      <td>W</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>Tottenham</td>
      <td>...</td>
      <td>Match Report</td>
      <td>NaN</td>
      <td>14.0</td>
      <td>7.0</td>
      <td>14.7</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2021</td>
      <td>Liverpool</td>
    </tr>
    <tr>
      <th>31</th>
      <td>2021-01-31</td>
      <td>16:30</td>
      <td>Premier League</td>
      <td>Matchweek 21</td>
      <td>Sun</td>
      <td>Away</td>
      <td>W</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>West Ham</td>
      <td>...</td>
      <td>Match Report</td>
      <td>NaN</td>
      <td>14.0</td>
      <td>5.0</td>
      <td>15.3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2021</td>
      <td>Liverpool</td>
    </tr>
    <tr>
      <th>32</th>
      <td>2021-02-03</td>
      <td>20:15</td>
      <td>Premier League</td>
      <td>Matchweek 22</td>
      <td>Wed</td>
      <td>Home</td>
      <td>L</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>Brighton</td>
      <td>...</td>
      <td>Match Report</td>
      <td>NaN</td>
      <td>11.0</td>
      <td>0.0</td>
      <td>19.9</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2021</td>
      <td>Liverpool</td>
    </tr>
    <tr>
      <th>33</th>
      <td>2021-02-07</td>
      <td>16:30</td>
      <td>Premier League</td>
      <td>Matchweek 23</td>
      <td>Sun</td>
      <td>Home</td>
      <td>L</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>Manchester City</td>
      <td>...</td>
      <td>Match Report</td>
      <td>NaN</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>17.9</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2021</td>
      <td>Liverpool</td>
    </tr>
    <tr>
      <th>34</th>
      <td>2021-02-13</td>
      <td>12:30</td>
      <td>Premier League</td>
      <td>Matchweek 24</td>
      <td>Sat</td>
      <td>Away</td>
      <td>L</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>Leicester City</td>
      <td>...</td>
      <td>Match Report</td>
      <td>NaN</td>
      <td>15.0</td>
      <td>4.0</td>
      <td>15.4</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2021</td>
      <td>Liverpool</td>
    </tr>
    <tr>
      <th>36</th>
      <td>2021-02-20</td>
      <td>17:30</td>
      <td>Premier League</td>
      <td>Matchweek 25</td>
      <td>Sat</td>
      <td>Home</td>
      <td>L</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>Everton</td>
      <td>...</td>
      <td>Match Report</td>
      <td>NaN</td>
      <td>16.0</td>
      <td>6.0</td>
      <td>15.9</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2021</td>
      <td>Liverpool</td>
    </tr>
    <tr>
      <th>37</th>
      <td>2021-02-28</td>
      <td>19:15</td>
      <td>Premier League</td>
      <td>Matchweek 26</td>
      <td>Sun</td>
      <td>Away</td>
      <td>W</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>Sheffield Utd</td>
      <td>...</td>
      <td>Match Report</td>
      <td>NaN</td>
      <td>15.0</td>
      <td>8.0</td>
      <td>14.2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2021</td>
      <td>Liverpool</td>
    </tr>
    <tr>
      <th>38</th>
      <td>2021-03-04</td>
      <td>20:15</td>
      <td>Premier League</td>
      <td>Matchweek 29</td>
      <td>Thu</td>
      <td>Home</td>
      <td>L</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>Chelsea</td>
      <td>...</td>
      <td>Match Report</td>
      <td>NaN</td>
      <td>7.0</td>
      <td>1.0</td>
      <td>18.4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2021</td>
      <td>Liverpool</td>
    </tr>
    <tr>
      <th>39</th>
      <td>2021-03-07</td>
      <td>14:00</td>
      <td>Premier League</td>
      <td>Matchweek 27</td>
      <td>Sun</td>
      <td>Home</td>
      <td>L</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>Fulham</td>
      <td>...</td>
      <td>Match Report</td>
      <td>NaN</td>
      <td>16.0</td>
      <td>3.0</td>
      <td>17.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2021</td>
      <td>Liverpool</td>
    </tr>
    <tr>
      <th>41</th>
      <td>2021-03-15</td>
      <td>20:00</td>
      <td>Premier League</td>
      <td>Matchweek 28</td>
      <td>Mon</td>
      <td>Away</td>
      <td>W</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>Wolves</td>
      <td>...</td>
      <td>Match Report</td>
      <td>NaN</td>
      <td>12.0</td>
      <td>4.0</td>
      <td>15.9</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2021</td>
      <td>Liverpool</td>
    </tr>
    <tr>
      <th>42</th>
      <td>2021-04-03</td>
      <td>20:00</td>
      <td>Premier League</td>
      <td>Matchweek 30</td>
      <td>Sat</td>
      <td>Away</td>
      <td>W</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>Arsenal</td>
      <td>...</td>
      <td>Match Report</td>
      <td>NaN</td>
      <td>16.0</td>
      <td>7.0</td>
      <td>17.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2021</td>
      <td>Liverpool</td>
    </tr>
    <tr>
      <th>44</th>
      <td>2021-04-10</td>
      <td>15:00</td>
      <td>Premier League</td>
      <td>Matchweek 31</td>
      <td>Sat</td>
      <td>Home</td>
      <td>W</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>Aston Villa</td>
      <td>...</td>
      <td>Match Report</td>
      <td>NaN</td>
      <td>23.0</td>
      <td>8.0</td>
      <td>16.7</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2021</td>
      <td>Liverpool</td>
    </tr>
    <tr>
      <th>46</th>
      <td>2021-04-19</td>
      <td>20:00</td>
      <td>Premier League</td>
      <td>Matchweek 32</td>
      <td>Mon</td>
      <td>Away</td>
      <td>D</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Leeds United</td>
      <td>...</td>
      <td>Match Report</td>
      <td>NaN</td>
      <td>17.0</td>
      <td>7.0</td>
      <td>15.7</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2021</td>
      <td>Liverpool</td>
    </tr>
    <tr>
      <th>47</th>
      <td>2021-04-24</td>
      <td>12:30</td>
      <td>Premier League</td>
      <td>Matchweek 33</td>
      <td>Sat</td>
      <td>Home</td>
      <td>D</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Newcastle Utd</td>
      <td>...</td>
      <td>Match Report</td>
      <td>NaN</td>
      <td>21.0</td>
      <td>9.0</td>
      <td>17.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2021</td>
      <td>Liverpool</td>
    </tr>
    <tr>
      <th>48</th>
      <td>2021-05-08</td>
      <td>20:15</td>
      <td>Premier League</td>
      <td>Matchweek 35</td>
      <td>Sat</td>
      <td>Home</td>
      <td>W</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>Southampton</td>
      <td>...</td>
      <td>Match Report</td>
      <td>NaN</td>
      <td>14.0</td>
      <td>6.0</td>
      <td>12.4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2021</td>
      <td>Liverpool</td>
    </tr>
    <tr>
      <th>49</th>
      <td>2021-05-13</td>
      <td>20:15</td>
      <td>Premier League</td>
      <td>Matchweek 34</td>
      <td>Thu</td>
      <td>Away</td>
      <td>W</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>Manchester Utd</td>
      <td>...</td>
      <td>Match Report</td>
      <td>NaN</td>
      <td>17.0</td>
      <td>8.0</td>
      <td>14.9</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2021</td>
      <td>Liverpool</td>
    </tr>
    <tr>
      <th>50</th>
      <td>2021-05-16</td>
      <td>16:30</td>
      <td>Premier League</td>
      <td>Matchweek 36</td>
      <td>Sun</td>
      <td>Away</td>
      <td>W</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>West Brom</td>
      <td>...</td>
      <td>Match Report</td>
      <td>NaN</td>
      <td>26.0</td>
      <td>6.0</td>
      <td>16.9</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2021</td>
      <td>Liverpool</td>
    </tr>
    <tr>
      <th>51</th>
      <td>2021-05-19</td>
      <td>20:15</td>
      <td>Premier League</td>
      <td>Matchweek 37</td>
      <td>Wed</td>
      <td>Away</td>
      <td>W</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>Burnley</td>
      <td>...</td>
      <td>Match Report</td>
      <td>NaN</td>
      <td>20.0</td>
      <td>3.0</td>
      <td>15.5</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2021</td>
      <td>Liverpool</td>
    </tr>
    <tr>
      <th>52</th>
      <td>2021-05-23</td>
      <td>16:00</td>
      <td>Premier League</td>
      <td>Matchweek 38</td>
      <td>Sun</td>
      <td>Home</td>
      <td>W</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>Crystal Palace</td>
      <td>...</td>
      <td>Match Report</td>
      <td>NaN</td>
      <td>19.0</td>
      <td>5.0</td>
      <td>14.2</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2021</td>
      <td>Liverpool</td>
    </tr>
  </tbody>
</table>
<p>38 rows × 27 columns</p>
</div>



It would appear that we only have data for the **2020/21** season for Liverpool, and we are missing data for the **2021/22** season. The data should still be fine to work with, and at least we have an explanation for why some of the data is missing.

Next, we will look at the variable **round**. This gives the Matchweek that each game was played in. In an EPL season there are **38 Matchweeks**.

For most match weeks, we would typically expect the count to be **40** for each Matchweek, as we count one matchweek per team, and we have **20 teams** playing in each matchweek, across **2 seasons**, thus $20 \times 2=40$. However, we are missing Liverpool's 2021/22 season, so the maximum number we would expect is a count of 39.

Also, this data was scraped mid-season for the 2021/22 season, thus some of the later matchweeks will not have been played yet.


```python
matches["round"].value_counts().to_frame()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>round</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Matchweek 1</th>
      <td>39</td>
    </tr>
    <tr>
      <th>Matchweek 16</th>
      <td>39</td>
    </tr>
    <tr>
      <th>Matchweek 34</th>
      <td>39</td>
    </tr>
    <tr>
      <th>Matchweek 32</th>
      <td>39</td>
    </tr>
    <tr>
      <th>Matchweek 31</th>
      <td>39</td>
    </tr>
    <tr>
      <th>Matchweek 29</th>
      <td>39</td>
    </tr>
    <tr>
      <th>Matchweek 28</th>
      <td>39</td>
    </tr>
    <tr>
      <th>Matchweek 26</th>
      <td>39</td>
    </tr>
    <tr>
      <th>Matchweek 25</th>
      <td>39</td>
    </tr>
    <tr>
      <th>Matchweek 24</th>
      <td>39</td>
    </tr>
    <tr>
      <th>Matchweek 23</th>
      <td>39</td>
    </tr>
    <tr>
      <th>Matchweek 2</th>
      <td>39</td>
    </tr>
    <tr>
      <th>Matchweek 19</th>
      <td>39</td>
    </tr>
    <tr>
      <th>Matchweek 17</th>
      <td>39</td>
    </tr>
    <tr>
      <th>Matchweek 20</th>
      <td>39</td>
    </tr>
    <tr>
      <th>Matchweek 15</th>
      <td>39</td>
    </tr>
    <tr>
      <th>Matchweek 5</th>
      <td>39</td>
    </tr>
    <tr>
      <th>Matchweek 3</th>
      <td>39</td>
    </tr>
    <tr>
      <th>Matchweek 13</th>
      <td>39</td>
    </tr>
    <tr>
      <th>Matchweek 12</th>
      <td>39</td>
    </tr>
    <tr>
      <th>Matchweek 4</th>
      <td>39</td>
    </tr>
    <tr>
      <th>Matchweek 11</th>
      <td>39</td>
    </tr>
    <tr>
      <th>Matchweek 10</th>
      <td>39</td>
    </tr>
    <tr>
      <th>Matchweek 9</th>
      <td>39</td>
    </tr>
    <tr>
      <th>Matchweek 8</th>
      <td>39</td>
    </tr>
    <tr>
      <th>Matchweek 14</th>
      <td>39</td>
    </tr>
    <tr>
      <th>Matchweek 7</th>
      <td>39</td>
    </tr>
    <tr>
      <th>Matchweek 6</th>
      <td>39</td>
    </tr>
    <tr>
      <th>Matchweek 30</th>
      <td>37</td>
    </tr>
    <tr>
      <th>Matchweek 27</th>
      <td>37</td>
    </tr>
    <tr>
      <th>Matchweek 22</th>
      <td>37</td>
    </tr>
    <tr>
      <th>Matchweek 21</th>
      <td>37</td>
    </tr>
    <tr>
      <th>Matchweek 18</th>
      <td>37</td>
    </tr>
    <tr>
      <th>Matchweek 33</th>
      <td>32</td>
    </tr>
    <tr>
      <th>Matchweek 35</th>
      <td>20</td>
    </tr>
    <tr>
      <th>Matchweek 36</th>
      <td>20</td>
    </tr>
    <tr>
      <th>Matchweek 37</th>
      <td>20</td>
    </tr>
    <tr>
      <th>Matchweek 38</th>
      <td>20</td>
    </tr>
  </tbody>
</table>
</div>



We can see that we have $39$ instances of each matchweek, as expected, but fewer counts for matchweeks $33$ onwards. This is because the last matchweek in the $2021/22$ season that had been played was the $32^{nd}$ matchweek. This is fine.

Now, we will check the data types of our variables using the **.dtypes** method. This is important as machine learning algorithms can only worth with numeric data.

If the column is stored as a different data type - such as *object* which typically denotes a string - then we have to find a way of converting these data types to numeric data in order to use them as predictors in our machine learning algorithm.


```python
type = matches.dtypes.to_frame().rename(columns={0:" Data Type"})
type
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Data Type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>date</th>
      <td>object</td>
    </tr>
    <tr>
      <th>time</th>
      <td>object</td>
    </tr>
    <tr>
      <th>comp</th>
      <td>object</td>
    </tr>
    <tr>
      <th>round</th>
      <td>object</td>
    </tr>
    <tr>
      <th>day</th>
      <td>object</td>
    </tr>
    <tr>
      <th>venue</th>
      <td>object</td>
    </tr>
    <tr>
      <th>result</th>
      <td>object</td>
    </tr>
    <tr>
      <th>gf</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>ga</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>opponent</th>
      <td>object</td>
    </tr>
    <tr>
      <th>xg</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>xga</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>poss</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>attendance</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>captain</th>
      <td>object</td>
    </tr>
    <tr>
      <th>formation</th>
      <td>object</td>
    </tr>
    <tr>
      <th>referee</th>
      <td>object</td>
    </tr>
    <tr>
      <th>match report</th>
      <td>object</td>
    </tr>
    <tr>
      <th>notes</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>sh</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>sot</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>dist</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>fk</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>pk</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>pkatt</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>season</th>
      <td>int64</td>
    </tr>
    <tr>
      <th>team</th>
      <td>object</td>
    </tr>
  </tbody>
</table>
</div>



### Remove Unnecessary Variables ⚽ <a class="anchor" id="2.8"></a>


```python
del matches["comp"]
```


```python
del matches["notes"]
```

### Create New Variables ⚽ <a class="anchor" id="2.9"></a>

The date variable appears to be stored as a string. We will over-write the existing column by converting it to a **datetime** data type using the **.to_datetime** method.

This will make it easier for us to use the date variable to compute predictors. For example, you could extract the month, or week day of the match.


```python
matches["date"] = pd.to_datetime(matches["date"])
```

We also require a **target** variable to denote whether or not the team won a match.


```python
matches["target"] = (matches["result"] == "W").astype("int")
```

The **venue_code** variable gives details on whether the team played at their home stadium, or at another team's stadium. This could be important, as the support of the home crowd can have a significant influence on the morale and performance of a team.

Currently, this variable is stored as a string, so we will convert it to a categorical data type using the **.astype( )** method. We will convert these categories to integers using the **.cat.codes** method. This will assign the value **0** for Away games and **1** for Home games.


```python
matches["venue_code"] = matches["venue"].astype("category").cat.codes
```

We will repeat this by assigning integer values to the **opponent** variable. Evidently, the opponent will influence the outcome of the match.


```python
matches["opp_code"] = matches["opponent"].astype("category").cat.codes
```

Do certain teams play better at certain times of day? To see if this may be the case, we will create a variable for the hour that the match is played.


```python
# Use a regular expression to extract the hour
matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype("int")
```

We will create one final variable that will return the day of the week that the match is played on, assuming that some teams may play better on - for example - a Sunday versus a Saturday.


```python
matches["day_code"] = matches["date"].dt.dayofweek
```


```python
# Display data frame
matches
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>time</th>
      <th>round</th>
      <th>day</th>
      <th>venue</th>
      <th>result</th>
      <th>gf</th>
      <th>ga</th>
      <th>opponent</th>
      <th>xg</th>
      <th>...</th>
      <th>fk</th>
      <th>pk</th>
      <th>pkatt</th>
      <th>season</th>
      <th>team</th>
      <th>target</th>
      <th>venue_code</th>
      <th>opp_code</th>
      <th>hour</th>
      <th>day_code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2021-08-15</td>
      <td>16:30</td>
      <td>Matchweek 1</td>
      <td>Sun</td>
      <td>Away</td>
      <td>L</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>Tottenham</td>
      <td>1.9</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2022</td>
      <td>Manchester City</td>
      <td>0</td>
      <td>0</td>
      <td>18</td>
      <td>16</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2021-08-21</td>
      <td>15:00</td>
      <td>Matchweek 2</td>
      <td>Sat</td>
      <td>Home</td>
      <td>W</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>Norwich City</td>
      <td>2.7</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2022</td>
      <td>Manchester City</td>
      <td>1</td>
      <td>1</td>
      <td>15</td>
      <td>15</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2021-08-28</td>
      <td>12:30</td>
      <td>Matchweek 3</td>
      <td>Sat</td>
      <td>Home</td>
      <td>W</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>Arsenal</td>
      <td>3.8</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2022</td>
      <td>Manchester City</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>12</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021-09-11</td>
      <td>15:00</td>
      <td>Matchweek 4</td>
      <td>Sat</td>
      <td>Away</td>
      <td>W</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>Leicester City</td>
      <td>2.9</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2022</td>
      <td>Manchester City</td>
      <td>1</td>
      <td>0</td>
      <td>10</td>
      <td>15</td>
      <td>5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2021-09-18</td>
      <td>15:00</td>
      <td>Matchweek 5</td>
      <td>Sat</td>
      <td>Home</td>
      <td>D</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Southampton</td>
      <td>1.1</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2022</td>
      <td>Manchester City</td>
      <td>0</td>
      <td>1</td>
      <td>17</td>
      <td>15</td>
      <td>5</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>38</th>
      <td>2021-05-02</td>
      <td>19:15</td>
      <td>Matchweek 34</td>
      <td>Sun</td>
      <td>Away</td>
      <td>L</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>Tottenham</td>
      <td>0.5</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2021</td>
      <td>Sheffield United</td>
      <td>0</td>
      <td>0</td>
      <td>18</td>
      <td>19</td>
      <td>6</td>
    </tr>
    <tr>
      <th>39</th>
      <td>2021-05-08</td>
      <td>15:00</td>
      <td>Matchweek 35</td>
      <td>Sat</td>
      <td>Home</td>
      <td>L</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>Crystal Palace</td>
      <td>0.7</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2021</td>
      <td>Sheffield United</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>15</td>
      <td>5</td>
    </tr>
    <tr>
      <th>40</th>
      <td>2021-05-16</td>
      <td>19:00</td>
      <td>Matchweek 36</td>
      <td>Sun</td>
      <td>Away</td>
      <td>W</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>Everton</td>
      <td>1.6</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2021</td>
      <td>Sheffield United</td>
      <td>1</td>
      <td>0</td>
      <td>7</td>
      <td>19</td>
      <td>6</td>
    </tr>
    <tr>
      <th>41</th>
      <td>2021-05-19</td>
      <td>18:00</td>
      <td>Matchweek 37</td>
      <td>Wed</td>
      <td>Away</td>
      <td>L</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>Newcastle Utd</td>
      <td>0.8</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2021</td>
      <td>Sheffield United</td>
      <td>0</td>
      <td>0</td>
      <td>14</td>
      <td>18</td>
      <td>2</td>
    </tr>
    <tr>
      <th>42</th>
      <td>2021-05-23</td>
      <td>16:00</td>
      <td>Matchweek 38</td>
      <td>Sun</td>
      <td>Home</td>
      <td>W</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>Burnley</td>
      <td>0.6</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2021</td>
      <td>Sheffield United</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>16</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
<p>1389 rows × 30 columns</p>
</div>



## Machine Learning ⚽ <a class="anchor" id="3"></a>

In this section, we will begin by training our machine learning model after giving a brief justification for the choice of algorithm: **Random Forest**.

After training our model, we will assess its predictive accuracy, and then seek to optimise this and improve the performance of our model.

![Random%20Forest.PNG](attachment:Random%20Forest.PNG)

### Random Forest ⚽ <a class="anchor" id="3.1"></a>

Random forest is a commonly-used machine learning algorithm, which combines the output of multiple **decision trees** to reach a single result via *majority voting*. 

While its ease of use and flexibility have lent it to both classification and regression problems, we will be using it for a **classification** problem in this project, i.e. classifying whether or not a team will win a match of football.

### Why Random Forest? ⚽ <a class="anchor" id="3.2"></a>

The Random Forest model is useful as it can pick up *non-linearities* in the data that some other algorithms may struggle with. 

For example, we have created a column for **opponent** and assigned codes to it. A code of 15, for example, does not denote a team that is more or less challenging to play against that a team with code 14 or less; they are just values to categorise different opponents numerically. A linear model would not be able to pick that up whereas a Random Forest can.

### Training Algorithm ⚽ <a class="anchor" id="3.3"></a>

First, we need to import the **RandomForest Classifier** from the **sklearn** library.


```python
from sklearn.ensemble import RandomForestClassifier
```

Now, we want to create a Random Forest instance by initialising the Random Forest Class. Below, we are going to enter in some hyper-paramters:

- **n_estimators** - How many Decision Trees we want in our Random Forest model
- **min_samples_split** - Number of samples we want to have in a leaf of a Decision Tree prior to splitting the node
- **random_state** - Set this to generate the same results each run


```python
rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)
```

The next step is to create a **training** data set and a **testing** data set.


```python
train = matches[matches["date"] < '2022-01-01']
```


```python
test = matches[matches["date"] > '2022-01-01']
```

Create a list of the predictor columns that we created earlier.


```python
predictors = ["venue_code", "opp_code", "hour", "day_code"]
```

Next, **fit** the model to the training data.


```python
rf.fit(train[predictors], train["target"])
```




    RandomForestClassifier(min_samples_split=10, n_estimators=50, random_state=1)



### Prediction & Accuracy ⚽ <a class="anchor" id="3.4"></a>

So far, we have initialised a Random Forest model and trained the model using our test data set.

Now, we generate **predictions** using the test data set to assess the performance of the model that we have just trained.


```python
preds = rf.predict(test[predictors])
```

To access the performance of our model, we will import the **accuracy_score** module.

Accuracy Score is a metric that returns a score for the **proportion of predictions made correctly**. For example, in this project, the number of 'Wins' and the number of 'Not Wins' that were correctly predicted are added together and divided by the total number of predictions made. If we made 6 correct predictions out of a total of 10 predictions, then our accuracy score would be: 

$$\frac{\text{# Correct Predictions}}{\text{# Total Predictions}} = \frac{6}{10}=0.6$$


```python
from sklearn.metrics import accuracy_score
```


```python
error = accuracy_score(test["target"], preds)
```


```python
error
```




    0.6123188405797102



A prediction **accuracy** of $\simeq 0.612$ suggests that our model correctly predicted the match result $61.2\%$ of the time.

We can go further, and drill down into this data to see if our model was better at - say - predicting wins versus predicting losses.

To do this, we will create a data frame that combines our actual values versus our predicted values. After this, we can create a **confusion matrix** (see below) using *cross-tabulation*.

![Confusion%20Matrix.PNG](attachment:Confusion%20Matrix.PNG)


```python
combined = pd.DataFrame(dict(actual=test["target"], predicted=preds))
```


```python
pd.crosstab(index=combined["actual"], columns=combined["predicted"])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>predicted</th>
      <th>0</th>
      <th>1</th>
    </tr>
    <tr>
      <th>actual</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>141</td>
      <td>31</td>
    </tr>
    <tr>
      <th>1</th>
      <td>76</td>
      <td>28</td>
    </tr>
  </tbody>
</table>
</div>



When we predicted a win, we were correct 28 times out of 59 (28 + 31) times, for example. This is approximately $47\%$ of the time, whereas we correctly predicted losses and draws approximately $65\%$ of the time.

It would appear that we are far **better at predicting losses and draws than wins**. Unfortunately, we are more concerned with out model predicting *wins* than losses and draws, so we will have to refine our model.

In light of this, we will revise our accuracy metric, and instead use the **precision score**.

The precision score tells us what proportion of the time we successfully predicted wins:

$$\text{Precision}=\frac{\text{True Positive}}{\text{True Positive + False Positive}}$$




```python
from sklearn.metrics import precision_score

precision_score(test["target"], preds)
```




    0.4745762711864407



This confirms our calculation above, that our precision is only around $47\%$. This isn't very good, so we are going to improve the model to see how this will affect its predictive ability.

### Improving the model ⚽ <a class="anchor" id="3.5"></a>

One way in which we can improve the model is to calculate how well a team had been performing going into a given match. Maybe if they were on a winning streak prior to a match, this may increase the likelihood they will win that match.

To create this variable, we will generate a **rolling average** of their match statistics.

To achieve this, we will create an object called *grouped_matches*. Essentially, this will create a separate data frame for each team in our data set.


```python
grouped_matches = matches.groupby("team")
```

After creating this object, we can select a particular team using the **.get_group( )** method. We will use Manchester City as an example.


```python
group = grouped_matches.get_group("Manchester City").sort_values("date")
```

This *group* object is a data frame containing all of the matches that Manchester City has played in.

We will create a rolling average function that will take in a group (team), a set of variables (match stats) from our existing data, and a set of new columns that the function will populate with the rolling averages for various match statistics.

We will want this function to:

1. Sort group in ascending order by date
2. Create a variable called rolling_stats that will take variables passed into the function and compute rolling averages
3. Assign rolling averages back to data frame as variables
4. Drop missing values


```python
def rolling_averages(group, cols, new_cols):
    
    # (1) - Sort group by date
    group = group.sort_values("date")
    
    # (2) - Create rolling averages
    rolling_stats = group[cols].rolling(3, closed='left').mean() # closed = left takes preceding values
    
    # (3) - Create new data frame columns for rolling averages
    group[new_cols] = rolling_stats
    
    # (4) - Drop NA
    group = group.dropna(subset=new_cols)
    
    return group
```

To create our new variables for our rolling averages to populate, we will use string formatting to add a 'rolling'-suffix to the names of existing variables, for example $$\text{sh}\Rightarrow \text{sh_rolling}$$


```python
cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]
new_cols = [f"{c}_rolling" for c in cols]

rolling_averages(group, cols, new_cols)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>time</th>
      <th>round</th>
      <th>day</th>
      <th>venue</th>
      <th>result</th>
      <th>gf</th>
      <th>ga</th>
      <th>opponent</th>
      <th>xg</th>
      <th>...</th>
      <th>hour</th>
      <th>day_code</th>
      <th>gf_rolling</th>
      <th>ga_rolling</th>
      <th>sh_rolling</th>
      <th>sot_rolling</th>
      <th>dist_rolling</th>
      <th>fk_rolling</th>
      <th>pk_rolling</th>
      <th>pkatt_rolling</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>2020-10-17</td>
      <td>17:30</td>
      <td>Matchweek 5</td>
      <td>Sat</td>
      <td>Home</td>
      <td>W</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>Arsenal</td>
      <td>1.5</td>
      <td>...</td>
      <td>17</td>
      <td>5</td>
      <td>2.000000</td>
      <td>2.333333</td>
      <td>17.333333</td>
      <td>4.666667</td>
      <td>18.900000</td>
      <td>1.333333</td>
      <td>0.333333</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2020-10-24</td>
      <td>12:30</td>
      <td>Matchweek 6</td>
      <td>Sat</td>
      <td>Away</td>
      <td>D</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>West Ham</td>
      <td>1.1</td>
      <td>...</td>
      <td>12</td>
      <td>5</td>
      <td>1.333333</td>
      <td>2.000000</td>
      <td>17.333333</td>
      <td>3.666667</td>
      <td>17.733333</td>
      <td>0.666667</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2020-10-31</td>
      <td>12:30</td>
      <td>Matchweek 7</td>
      <td>Sat</td>
      <td>Away</td>
      <td>W</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>Sheffield Utd</td>
      <td>1.5</td>
      <td>...</td>
      <td>12</td>
      <td>5</td>
      <td>1.000000</td>
      <td>0.666667</td>
      <td>16.666667</td>
      <td>4.333333</td>
      <td>18.233333</td>
      <td>0.666667</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2020-11-08</td>
      <td>16:30</td>
      <td>Matchweek 8</td>
      <td>Sun</td>
      <td>Home</td>
      <td>D</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Liverpool</td>
      <td>1.6</td>
      <td>...</td>
      <td>16</td>
      <td>6</td>
      <td>1.000000</td>
      <td>0.333333</td>
      <td>14.333333</td>
      <td>6.666667</td>
      <td>18.466667</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2020-11-21</td>
      <td>17:30</td>
      <td>Matchweek 9</td>
      <td>Sat</td>
      <td>Away</td>
      <td>L</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>Tottenham</td>
      <td>1.3</td>
      <td>...</td>
      <td>17</td>
      <td>5</td>
      <td>1.000000</td>
      <td>0.666667</td>
      <td>12.000000</td>
      <td>5.666667</td>
      <td>19.366667</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>42</th>
      <td>2022-03-14</td>
      <td>20:00</td>
      <td>Matchweek 29</td>
      <td>Mon</td>
      <td>Away</td>
      <td>D</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Crystal Palace</td>
      <td>2.3</td>
      <td>...</td>
      <td>20</td>
      <td>0</td>
      <td>2.333333</td>
      <td>1.333333</td>
      <td>19.000000</td>
      <td>7.000000</td>
      <td>15.366667</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>44</th>
      <td>2022-04-02</td>
      <td>15:00</td>
      <td>Matchweek 31</td>
      <td>Sat</td>
      <td>Away</td>
      <td>W</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>Burnley</td>
      <td>1.8</td>
      <td>...</td>
      <td>15</td>
      <td>5</td>
      <td>1.666667</td>
      <td>0.333333</td>
      <td>18.333333</td>
      <td>7.333333</td>
      <td>16.000000</td>
      <td>0.333333</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>46</th>
      <td>2022-04-10</td>
      <td>16:30</td>
      <td>Matchweek 32</td>
      <td>Sun</td>
      <td>Home</td>
      <td>D</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>Liverpool</td>
      <td>2.0</td>
      <td>...</td>
      <td>16</td>
      <td>6</td>
      <td>2.000000</td>
      <td>0.333333</td>
      <td>20.000000</td>
      <td>6.666667</td>
      <td>16.133333</td>
      <td>0.333333</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>49</th>
      <td>2022-04-20</td>
      <td>20:00</td>
      <td>Matchweek 30</td>
      <td>Wed</td>
      <td>Home</td>
      <td>W</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>Brighton</td>
      <td>1.2</td>
      <td>...</td>
      <td>20</td>
      <td>2</td>
      <td>1.333333</td>
      <td>0.666667</td>
      <td>15.666667</td>
      <td>4.666667</td>
      <td>16.700000</td>
      <td>0.333333</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50</th>
      <td>2022-04-23</td>
      <td>15:00</td>
      <td>Matchweek 34</td>
      <td>Sat</td>
      <td>Home</td>
      <td>W</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>Watford</td>
      <td>3.0</td>
      <td>...</td>
      <td>15</td>
      <td>5</td>
      <td>2.333333</td>
      <td>0.666667</td>
      <td>15.333333</td>
      <td>5.000000</td>
      <td>17.200000</td>
      <td>0.333333</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>68 rows × 38 columns</p>
</div>



Now we have shown this for Manchester City, we will create a **lambda function** and the **.apply( )** method to iterate this function over *all* of the teams in our data set. Very cool!


```python
matches_rolling = matches.groupby("team").apply(lambda x: rolling_averages(x, cols, new_cols))
```


```python
matches_rolling
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>date</th>
      <th>time</th>
      <th>round</th>
      <th>day</th>
      <th>venue</th>
      <th>result</th>
      <th>gf</th>
      <th>ga</th>
      <th>opponent</th>
      <th>xg</th>
      <th>...</th>
      <th>hour</th>
      <th>day_code</th>
      <th>gf_rolling</th>
      <th>ga_rolling</th>
      <th>sh_rolling</th>
      <th>sot_rolling</th>
      <th>dist_rolling</th>
      <th>fk_rolling</th>
      <th>pk_rolling</th>
      <th>pkatt_rolling</th>
    </tr>
    <tr>
      <th>team</th>
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
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">Arsenal</th>
      <th>6</th>
      <td>2020-10-04</td>
      <td>14:00</td>
      <td>Matchweek 4</td>
      <td>Sun</td>
      <td>Home</td>
      <td>W</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>Sheffield Utd</td>
      <td>0.4</td>
      <td>...</td>
      <td>14</td>
      <td>6</td>
      <td>2.000000</td>
      <td>1.333333</td>
      <td>7.666667</td>
      <td>3.666667</td>
      <td>14.733333</td>
      <td>0.666667</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2020-10-17</td>
      <td>17:30</td>
      <td>Matchweek 5</td>
      <td>Sat</td>
      <td>Away</td>
      <td>L</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>Manchester City</td>
      <td>0.9</td>
      <td>...</td>
      <td>17</td>
      <td>5</td>
      <td>1.666667</td>
      <td>1.666667</td>
      <td>5.333333</td>
      <td>3.666667</td>
      <td>15.766667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2020-10-25</td>
      <td>19:15</td>
      <td>Matchweek 6</td>
      <td>Sun</td>
      <td>Home</td>
      <td>L</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>Leicester City</td>
      <td>0.9</td>
      <td>...</td>
      <td>19</td>
      <td>6</td>
      <td>1.000000</td>
      <td>1.666667</td>
      <td>7.000000</td>
      <td>3.666667</td>
      <td>16.733333</td>
      <td>0.666667</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2020-11-01</td>
      <td>16:30</td>
      <td>Matchweek 7</td>
      <td>Sun</td>
      <td>Away</td>
      <td>W</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>Manchester Utd</td>
      <td>1.1</td>
      <td>...</td>
      <td>16</td>
      <td>6</td>
      <td>0.666667</td>
      <td>1.000000</td>
      <td>9.666667</td>
      <td>4.000000</td>
      <td>16.033333</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2020-11-08</td>
      <td>19:15</td>
      <td>Matchweek 8</td>
      <td>Sun</td>
      <td>Home</td>
      <td>L</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>Aston Villa</td>
      <td>1.5</td>
      <td>...</td>
      <td>19</td>
      <td>6</td>
      <td>0.333333</td>
      <td>0.666667</td>
      <td>9.666667</td>
      <td>2.666667</td>
      <td>18.033333</td>
      <td>1.000000</td>
      <td>0.333333</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">Wolverhampton Wanderers</th>
      <th>32</th>
      <td>2022-03-13</td>
      <td>14:00</td>
      <td>Matchweek 29</td>
      <td>Sun</td>
      <td>Away</td>
      <td>W</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>Everton</td>
      <td>0.8</td>
      <td>...</td>
      <td>14</td>
      <td>6</td>
      <td>1.333333</td>
      <td>1.000000</td>
      <td>12.333333</td>
      <td>3.666667</td>
      <td>19.300000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>33</th>
      <td>2022-03-18</td>
      <td>20:00</td>
      <td>Matchweek 30</td>
      <td>Fri</td>
      <td>Home</td>
      <td>L</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>Leeds United</td>
      <td>0.8</td>
      <td>...</td>
      <td>20</td>
      <td>4</td>
      <td>1.666667</td>
      <td>0.666667</td>
      <td>12.333333</td>
      <td>4.333333</td>
      <td>19.600000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>34</th>
      <td>2022-04-02</td>
      <td>15:00</td>
      <td>Matchweek 31</td>
      <td>Sat</td>
      <td>Home</td>
      <td>W</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>Aston Villa</td>
      <td>1.2</td>
      <td>...</td>
      <td>15</td>
      <td>5</td>
      <td>2.333333</td>
      <td>1.000000</td>
      <td>13.000000</td>
      <td>5.333333</td>
      <td>19.833333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>35</th>
      <td>2022-04-08</td>
      <td>20:00</td>
      <td>Matchweek 32</td>
      <td>Fri</td>
      <td>Away</td>
      <td>L</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>Newcastle Utd</td>
      <td>0.3</td>
      <td>...</td>
      <td>20</td>
      <td>4</td>
      <td>1.666667</td>
      <td>1.333333</td>
      <td>13.000000</td>
      <td>5.000000</td>
      <td>18.533333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>36</th>
      <td>2022-04-24</td>
      <td>14:00</td>
      <td>Matchweek 34</td>
      <td>Sun</td>
      <td>Away</td>
      <td>L</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>Burnley</td>
      <td>0.7</td>
      <td>...</td>
      <td>14</td>
      <td>6</td>
      <td>1.333333</td>
      <td>1.666667</td>
      <td>10.000000</td>
      <td>4.666667</td>
      <td>17.633333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>1317 rows × 38 columns</p>
</div>



Notice again that we appear to have generated an additional index level. Let's drop this before proceeding.


```python
matches_rolling = matches_rolling.droplevel('team')
```


```python
# Display data frame
matches_rolling
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>time</th>
      <th>round</th>
      <th>day</th>
      <th>venue</th>
      <th>result</th>
      <th>gf</th>
      <th>ga</th>
      <th>opponent</th>
      <th>xg</th>
      <th>...</th>
      <th>hour</th>
      <th>day_code</th>
      <th>gf_rolling</th>
      <th>ga_rolling</th>
      <th>sh_rolling</th>
      <th>sot_rolling</th>
      <th>dist_rolling</th>
      <th>fk_rolling</th>
      <th>pk_rolling</th>
      <th>pkatt_rolling</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>2020-10-04</td>
      <td>14:00</td>
      <td>Matchweek 4</td>
      <td>Sun</td>
      <td>Home</td>
      <td>W</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>Sheffield Utd</td>
      <td>0.4</td>
      <td>...</td>
      <td>14</td>
      <td>6</td>
      <td>2.000000</td>
      <td>1.333333</td>
      <td>7.666667</td>
      <td>3.666667</td>
      <td>14.733333</td>
      <td>0.666667</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2020-10-17</td>
      <td>17:30</td>
      <td>Matchweek 5</td>
      <td>Sat</td>
      <td>Away</td>
      <td>L</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>Manchester City</td>
      <td>0.9</td>
      <td>...</td>
      <td>17</td>
      <td>5</td>
      <td>1.666667</td>
      <td>1.666667</td>
      <td>5.333333</td>
      <td>3.666667</td>
      <td>15.766667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2020-10-25</td>
      <td>19:15</td>
      <td>Matchweek 6</td>
      <td>Sun</td>
      <td>Home</td>
      <td>L</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>Leicester City</td>
      <td>0.9</td>
      <td>...</td>
      <td>19</td>
      <td>6</td>
      <td>1.000000</td>
      <td>1.666667</td>
      <td>7.000000</td>
      <td>3.666667</td>
      <td>16.733333</td>
      <td>0.666667</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2020-11-01</td>
      <td>16:30</td>
      <td>Matchweek 7</td>
      <td>Sun</td>
      <td>Away</td>
      <td>W</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>Manchester Utd</td>
      <td>1.1</td>
      <td>...</td>
      <td>16</td>
      <td>6</td>
      <td>0.666667</td>
      <td>1.000000</td>
      <td>9.666667</td>
      <td>4.000000</td>
      <td>16.033333</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2020-11-08</td>
      <td>19:15</td>
      <td>Matchweek 8</td>
      <td>Sun</td>
      <td>Home</td>
      <td>L</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>Aston Villa</td>
      <td>1.5</td>
      <td>...</td>
      <td>19</td>
      <td>6</td>
      <td>0.333333</td>
      <td>0.666667</td>
      <td>9.666667</td>
      <td>2.666667</td>
      <td>18.033333</td>
      <td>1.000000</td>
      <td>0.333333</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>32</th>
      <td>2022-03-13</td>
      <td>14:00</td>
      <td>Matchweek 29</td>
      <td>Sun</td>
      <td>Away</td>
      <td>W</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>Everton</td>
      <td>0.8</td>
      <td>...</td>
      <td>14</td>
      <td>6</td>
      <td>1.333333</td>
      <td>1.000000</td>
      <td>12.333333</td>
      <td>3.666667</td>
      <td>19.300000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>33</th>
      <td>2022-03-18</td>
      <td>20:00</td>
      <td>Matchweek 30</td>
      <td>Fri</td>
      <td>Home</td>
      <td>L</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>Leeds United</td>
      <td>0.8</td>
      <td>...</td>
      <td>20</td>
      <td>4</td>
      <td>1.666667</td>
      <td>0.666667</td>
      <td>12.333333</td>
      <td>4.333333</td>
      <td>19.600000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>34</th>
      <td>2022-04-02</td>
      <td>15:00</td>
      <td>Matchweek 31</td>
      <td>Sat</td>
      <td>Home</td>
      <td>W</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>Aston Villa</td>
      <td>1.2</td>
      <td>...</td>
      <td>15</td>
      <td>5</td>
      <td>2.333333</td>
      <td>1.000000</td>
      <td>13.000000</td>
      <td>5.333333</td>
      <td>19.833333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>35</th>
      <td>2022-04-08</td>
      <td>20:00</td>
      <td>Matchweek 32</td>
      <td>Fri</td>
      <td>Away</td>
      <td>L</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>Newcastle Utd</td>
      <td>0.3</td>
      <td>...</td>
      <td>20</td>
      <td>4</td>
      <td>1.666667</td>
      <td>1.333333</td>
      <td>13.000000</td>
      <td>5.000000</td>
      <td>18.533333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>36</th>
      <td>2022-04-24</td>
      <td>14:00</td>
      <td>Matchweek 34</td>
      <td>Sun</td>
      <td>Away</td>
      <td>L</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>Burnley</td>
      <td>0.7</td>
      <td>...</td>
      <td>14</td>
      <td>6</td>
      <td>1.333333</td>
      <td>1.666667</td>
      <td>10.000000</td>
      <td>4.666667</td>
      <td>17.633333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>1317 rows × 38 columns</p>
</div>



It may be noticed that many of our index values are being repeated, so we will reset out index. This is important as we want unique values for our index.


```python
matches_rolling.index = range(matches_rolling.shape[0])
```

### Predictions Function ⚽ <a class="anchor" id="3.6"></a>

In this sub-section we will create a predictions function that we can use to make predictions without having to repeat significant chunks of code for each model we wish to test. This function will:

1. Split data into training- and test-sets
2. Fit the model to training data
3. Make predictions using testing data
4. Combine actuals and predictions together
5. Calculate precision


```python
def make_predictions(data, predictors):
    
    # (1) - Split data
    train = data[data["date"] < '2022-01-01']
    test = data[data["date"] > '2022-01-01']
    
    # (2) - Fit model
    rf.fit(train[predictors], train["target"])
    
    # (3) - Make predictions
    preds = rf.predict(test[predictors])
    
    # (4) - Combine actuals and predictions
    combined = pd.DataFrame(dict(actual=test["target"], predicted=preds), index=test.index)
    
    # (5) - Calculate precision
    error = precision_score(test["target"], preds)
    
    return combined, error
```

We can now call this function and pass in our original predictors *and* the rolling averages we have just generated.


```python
combined, error = make_predictions(matches_rolling, predictors + new_cols)
```


```python
error
```




    0.625



At $62.5\%$, we have clearly improved our precision markedly.

We can also see how the predicted result for each match compared with its actual result by running the *combined* command.


```python
combined
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>actual</th>
      <th>predicted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>55</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>56</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>57</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>58</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>59</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1312</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1313</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1314</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1315</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1316</th>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>276 rows × 2 columns</p>
</div>



However, this does not really give us much information about those matches, such as when the match happened, which teams were playing, etc. We can enrich our combined data frame using the **.merge( )** method to add in match details, as shown below.


```python
combined = combined.merge(matches_rolling[["date", "team", "opponent", "result"]], left_index=True, right_index=True)
```


```python
# Display first 10 matches
combined.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>actual</th>
      <th>predicted</th>
      <th>date</th>
      <th>team</th>
      <th>opponent</th>
      <th>result</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>55</th>
      <td>0</td>
      <td>0</td>
      <td>2022-01-23</td>
      <td>Arsenal</td>
      <td>Burnley</td>
      <td>D</td>
    </tr>
    <tr>
      <th>56</th>
      <td>1</td>
      <td>0</td>
      <td>2022-02-10</td>
      <td>Arsenal</td>
      <td>Wolves</td>
      <td>W</td>
    </tr>
    <tr>
      <th>57</th>
      <td>1</td>
      <td>0</td>
      <td>2022-02-19</td>
      <td>Arsenal</td>
      <td>Brentford</td>
      <td>W</td>
    </tr>
    <tr>
      <th>58</th>
      <td>1</td>
      <td>1</td>
      <td>2022-02-24</td>
      <td>Arsenal</td>
      <td>Wolves</td>
      <td>W</td>
    </tr>
    <tr>
      <th>59</th>
      <td>1</td>
      <td>1</td>
      <td>2022-03-06</td>
      <td>Arsenal</td>
      <td>Watford</td>
      <td>W</td>
    </tr>
    <tr>
      <th>60</th>
      <td>1</td>
      <td>1</td>
      <td>2022-03-13</td>
      <td>Arsenal</td>
      <td>Leicester City</td>
      <td>W</td>
    </tr>
    <tr>
      <th>61</th>
      <td>0</td>
      <td>1</td>
      <td>2022-03-16</td>
      <td>Arsenal</td>
      <td>Liverpool</td>
      <td>L</td>
    </tr>
    <tr>
      <th>62</th>
      <td>1</td>
      <td>0</td>
      <td>2022-03-19</td>
      <td>Arsenal</td>
      <td>Aston Villa</td>
      <td>W</td>
    </tr>
    <tr>
      <th>63</th>
      <td>0</td>
      <td>0</td>
      <td>2022-04-04</td>
      <td>Arsenal</td>
      <td>Crystal Palace</td>
      <td>L</td>
    </tr>
    <tr>
      <th>64</th>
      <td>0</td>
      <td>0</td>
      <td>2022-04-09</td>
      <td>Arsenal</td>
      <td>Brighton</td>
      <td>L</td>
    </tr>
  </tbody>
</table>
</div>



### Matching Results ⚽ <a class="anchor" id="3.7"></a>

In our analysis, we made predictions for each match **twice**. Once from the perspective of the Home team and one from the perspective of the Away team. **Did our model predict the same overall result for that match for both perspectives?** 

For example, if Arsenal is playing at Home against Everton in a given season, and the prediction is a win for Arsenal, then will the model also predict a loss or draw for Everton playing Away against Arsenal in the same season?

To check this, we can combine our data together. First, we need to ensure that the *team* name and the *opponent* name are the same in our data set, as sometimes they are not. For instance, we have **Wolverhampton Wanderers** in the *team* column and **Wolves** in the *opponent* column for the same team.

To make sure that the names are consistent across both columns, we will create a *dictionary* and use the **.map( )** method with that dictionary. However, we must first create a **child class** that *inherents* from the dictionary class. For a good explanation of class inheritance, click [here]("https://www.w3schools.com/python/python_inheritance.asp").

We need to do this because the .map( ) method does not handle any **missing keys**. So, if the dictionary being mapped to a column is - for example - missing a team name, the .map( ) method will simply remove that team's observation from the data. For instance, if we create a mapping dictionary that has a key 'Wolverhampton Wanderers' with a value 'Wolves', then passing 'Wolverhampton Wanderers' will return the value 'Wolves'. However, if we pass in 'Arsenal' but there is no key for this team, then the observation will be removed from the data. 

Instead, we want the mapping function to simply return the team name that it is passed if no key exists for that team. This is illustrated below.

![Mapping.PNG](attachment:Mapping.PNG)

We begin by creating a child class called **MissingDict**. We pass *dict* into the argument, so it will inherit from the *dict class*. This class will leverage a Python **hook** called *missing*. A hook can be used to tap into a module and react when something happens; in this case, when there is a missing key.  

After this, we want to create a mapping dictionary before finally passing this dictionary as an argument to our **MissingDict** class to create an instance of it, *mapping*. When we do this, we will use type two astericks prior to *map_values*. We do this because you cannot directly send a dictionary as a parameter to a function accepting kwargs. The **dictionary must be unpacked** so that the function may make use of its elements. The two astericks accomplish this.


```python
class MissingDict(dict):
    __missing__ = lambda self, key: key

map_values = {"Brighton and Hove Albion": "Brighton", 
              "Manchester United": "Manchester Utd", 
              "Newcastle United": "Newcastle Utd", 
              "Tottenham Hotspur": "Tottenham", 
              "West Ham United": "West Ham", 
              "Wolverhampton Wanderers": "Wolves"} 
mapping = MissingDict(**map_values)
```

We can now use this in our **.map( )** method.


```python
combined["new_team"] = combined["team"].map(mapping)

#Display data frame
combined
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>actual</th>
      <th>predicted</th>
      <th>date</th>
      <th>team</th>
      <th>opponent</th>
      <th>result</th>
      <th>new_team</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>55</th>
      <td>0</td>
      <td>0</td>
      <td>2022-01-23</td>
      <td>Arsenal</td>
      <td>Burnley</td>
      <td>D</td>
      <td>Arsenal</td>
    </tr>
    <tr>
      <th>56</th>
      <td>1</td>
      <td>0</td>
      <td>2022-02-10</td>
      <td>Arsenal</td>
      <td>Wolves</td>
      <td>W</td>
      <td>Arsenal</td>
    </tr>
    <tr>
      <th>57</th>
      <td>1</td>
      <td>0</td>
      <td>2022-02-19</td>
      <td>Arsenal</td>
      <td>Brentford</td>
      <td>W</td>
      <td>Arsenal</td>
    </tr>
    <tr>
      <th>58</th>
      <td>1</td>
      <td>1</td>
      <td>2022-02-24</td>
      <td>Arsenal</td>
      <td>Wolves</td>
      <td>W</td>
      <td>Arsenal</td>
    </tr>
    <tr>
      <th>59</th>
      <td>1</td>
      <td>1</td>
      <td>2022-03-06</td>
      <td>Arsenal</td>
      <td>Watford</td>
      <td>W</td>
      <td>Arsenal</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1312</th>
      <td>1</td>
      <td>0</td>
      <td>2022-03-13</td>
      <td>Wolverhampton Wanderers</td>
      <td>Everton</td>
      <td>W</td>
      <td>Wolves</td>
    </tr>
    <tr>
      <th>1313</th>
      <td>0</td>
      <td>0</td>
      <td>2022-03-18</td>
      <td>Wolverhampton Wanderers</td>
      <td>Leeds United</td>
      <td>L</td>
      <td>Wolves</td>
    </tr>
    <tr>
      <th>1314</th>
      <td>1</td>
      <td>0</td>
      <td>2022-04-02</td>
      <td>Wolverhampton Wanderers</td>
      <td>Aston Villa</td>
      <td>W</td>
      <td>Wolves</td>
    </tr>
    <tr>
      <th>1315</th>
      <td>0</td>
      <td>0</td>
      <td>2022-04-08</td>
      <td>Wolverhampton Wanderers</td>
      <td>Newcastle Utd</td>
      <td>L</td>
      <td>Wolves</td>
    </tr>
    <tr>
      <th>1316</th>
      <td>0</td>
      <td>0</td>
      <td>2022-04-24</td>
      <td>Wolverhampton Wanderers</td>
      <td>Burnley</td>
      <td>L</td>
      <td>Wolves</td>
    </tr>
  </tbody>
</table>
<p>276 rows × 7 columns</p>
</div>



As you can see above, we now have a **new_team** variable that contains the same teams as in the **team** column, but with the same naming convention as in the **opponent** column.

We can now merge this data frame *with itself*, by merging on the *date* and *new_team* variables for one data frame, but using the *date* and *opponent* in the other data frame. The purpose of this is to create a data frame with **two predicted** variables, one prediction made from the Home team's perspective, and the other from the Away team's perspective.

This process is illustrated below. **prediction_x** is the original match prediction made for that team, whereas **prediction_y** is the prediction from the perspective of the opponent.

![Merge.PNG](attachment:Merge.PNG)

In the first row in the top table in our illustration, Wolverhampton Wanderers are playing Arsenal at Home on the $1^{st}$ February $2022$. The predicted result for Wolverhampton Wanderers is a loss (L). The second row is the *same* match but from the perspective of Arsenal, who are playing Burnley Away on the $1^{st}$ February $2022$ and predicted to win (W). The third and fourth row relate to a single game between Aston Villa and Burnley.

The second table is post-merge. In the first row, we again have Wolverhampton Wanderers playing Arsenal at Home on the $1^{st}$ February $2022$. The **prediction_x** returns the original predicted match result for this team, a loss (L). **prediction_y** returns the predicted result from Arsenal's perspective, which is a win (W). This makes sense, as if one team loses, the other team must - by definition - win: these two columns should have *different values*.

For the Aston Villa and Burnley match, however, *prediction_x* and *prediction_y* are the same for both teams meaning that our algorithm predicted that *both* teams won the match. This is not possible in reality, so this would count against our model.


```python
merged = combined.merge(combined, left_on=["date", "new_team"], right_on=["date", "opponent"])
```


```python
merged
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>actual_x</th>
      <th>predicted_x</th>
      <th>date</th>
      <th>team_x</th>
      <th>opponent_x</th>
      <th>result_x</th>
      <th>new_team_x</th>
      <th>actual_y</th>
      <th>predicted_y</th>
      <th>team_y</th>
      <th>opponent_y</th>
      <th>result_y</th>
      <th>new_team_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>2022-01-23</td>
      <td>Arsenal</td>
      <td>Burnley</td>
      <td>D</td>
      <td>Arsenal</td>
      <td>0</td>
      <td>0</td>
      <td>Burnley</td>
      <td>Arsenal</td>
      <td>D</td>
      <td>Burnley</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>2022-02-10</td>
      <td>Arsenal</td>
      <td>Wolves</td>
      <td>W</td>
      <td>Arsenal</td>
      <td>0</td>
      <td>0</td>
      <td>Wolverhampton Wanderers</td>
      <td>Arsenal</td>
      <td>L</td>
      <td>Wolves</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>2022-02-19</td>
      <td>Arsenal</td>
      <td>Brentford</td>
      <td>W</td>
      <td>Arsenal</td>
      <td>0</td>
      <td>0</td>
      <td>Brentford</td>
      <td>Arsenal</td>
      <td>L</td>
      <td>Brentford</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>2022-02-24</td>
      <td>Arsenal</td>
      <td>Wolves</td>
      <td>W</td>
      <td>Arsenal</td>
      <td>0</td>
      <td>0</td>
      <td>Wolverhampton Wanderers</td>
      <td>Arsenal</td>
      <td>L</td>
      <td>Wolves</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>2022-03-06</td>
      <td>Arsenal</td>
      <td>Watford</td>
      <td>W</td>
      <td>Arsenal</td>
      <td>0</td>
      <td>0</td>
      <td>Watford</td>
      <td>Arsenal</td>
      <td>L</td>
      <td>Watford</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>257</th>
      <td>1</td>
      <td>0</td>
      <td>2022-03-13</td>
      <td>Wolverhampton Wanderers</td>
      <td>Everton</td>
      <td>W</td>
      <td>Wolves</td>
      <td>0</td>
      <td>0</td>
      <td>Everton</td>
      <td>Wolves</td>
      <td>L</td>
      <td>Everton</td>
    </tr>
    <tr>
      <th>258</th>
      <td>0</td>
      <td>0</td>
      <td>2022-03-18</td>
      <td>Wolverhampton Wanderers</td>
      <td>Leeds United</td>
      <td>L</td>
      <td>Wolves</td>
      <td>1</td>
      <td>0</td>
      <td>Leeds United</td>
      <td>Wolves</td>
      <td>W</td>
      <td>Leeds United</td>
    </tr>
    <tr>
      <th>259</th>
      <td>1</td>
      <td>0</td>
      <td>2022-04-02</td>
      <td>Wolverhampton Wanderers</td>
      <td>Aston Villa</td>
      <td>W</td>
      <td>Wolves</td>
      <td>0</td>
      <td>0</td>
      <td>Aston Villa</td>
      <td>Wolves</td>
      <td>L</td>
      <td>Aston Villa</td>
    </tr>
    <tr>
      <th>260</th>
      <td>0</td>
      <td>0</td>
      <td>2022-04-08</td>
      <td>Wolverhampton Wanderers</td>
      <td>Newcastle Utd</td>
      <td>L</td>
      <td>Wolves</td>
      <td>1</td>
      <td>0</td>
      <td>Newcastle United</td>
      <td>Wolves</td>
      <td>W</td>
      <td>Newcastle Utd</td>
    </tr>
    <tr>
      <th>261</th>
      <td>0</td>
      <td>0</td>
      <td>2022-04-24</td>
      <td>Wolverhampton Wanderers</td>
      <td>Burnley</td>
      <td>L</td>
      <td>Wolves</td>
      <td>1</td>
      <td>0</td>
      <td>Burnley</td>
      <td>Wolves</td>
      <td>W</td>
      <td>Burnley</td>
    </tr>
  </tbody>
</table>
<p>262 rows × 13 columns</p>
</div>



We can now look at just the rows where one team was predicted to win and where the other team was predicted to lose.


```python
merged[(merged["predicted_x"] == 1) & (merged["predicted_y"] ==0)]["actual_x"].value_counts()
```




    1    27
    0    13
    Name: actual_x, dtype: int64



This result suggests that our when our algorithm predicted that one team would win, it correctly predicted that the other team would lose $27$ out of $40$ times, which is $67.5\%$. This is not too terrible, but not great either.

## Conclusion ⚽ <a class="anchor" id="4"></a>

In this project, we web-scraped football data for English Premier League matches across two seasons. After cleaning our data, we enriched it by creating new predictors.

We used a Random Forest algorithm courtesy of the *sklearn* library. After training our model, we assessed its performance. Initially, this was not very strong with a precision of only $47\%$.

We further enriched our data by creating rolling averages for each team's performance in the prior three games. This resulted in a considerable improvement to our model, with a new precision score of $62.5\%$.

Finally, we checked the extent to which our made consistent predictions. We found that when our model predicted that a team would win a match, it also predicted that their opponent would lose $67.5\%$ of the time.

### Further Development Suggestions ⚽ <a class="anchor" id="4.1"></a>

There is a variety of ways in which this project could be improved upon. One way would be to change the values of our Random Forest **hyper-parameters**. We could, for instance, build models with greater or fewer Decision Trees. Typically, the more trees used, the greater the predictive power. This may lead to a higher computational workload, but moderate increases in the number of trees should not be a problem.

We could have **used more variables** from our data as predictors. For example, both the manager of the team and their opponent may influence the result. A manager of the calibre of Pep Guardiola may increase the likelihood that the team may win. That said, it is unlikely that a manager moves from one team to another team in the same league in such a small window of time, using the manager as a predictor not improve the model significantly over simply using the opponent code, as we did. However, employing other variables such as match attendance may prove fruitful.

Another suggestion would be to **web-scrape more match statistics**. On each team's webpage, there are other tabs aside from *Shooting* that we could use. A major drawback of this, however, is that it would require significantly more requests being sent to the web-server, increasing the risk of your machine being blocked.
