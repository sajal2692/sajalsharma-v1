---
layout: notebook
title: Moneyball - Using EDA to Identify Replacement Players
skills: R, Exploratory Data Analysis, gplot, dplyr
external_type: RPubs
external_url: http://rpubs.com/sajal_sharma/moneyball_lost_players
description: Exploration of baseball data for the year 2001 using R to look at replacements for key players lost by the Oakland A's in 2001.
---
---

[Background](https://en.wikipedia.org/wiki/Moneyball)

During the 2001-02 offseason, the Oakland A's team lost three key
players to teams with larger revenues.

The goal of this project is to look at player and salary data for those
years, to find players of the same calibre (statistically) who have been
undervalued by the market and thus, are suitable low salary
replacements.

[Data Source](http://www.seanlahman.com/baseball-archive/statistics/)

Setup
-----

    library(dplyr)
    library(ggplot2)

    batting <- read.csv('Batting.csv')
    sal <- read.csv('Salaries.csv')

    head(batting)

    ##    playerID yearID stint teamID lgID  G G_batting AB R H X2B X3B HR RBI SB
    ## 1 aardsda01   2004     1    SFN   NL 11        11  0 0 0   0   0  0   0  0
    ## 2 aardsda01   2006     1    CHN   NL 45        43  2 0 0   0   0  0   0  0
    ## 3 aardsda01   2007     1    CHA   AL 25         2  0 0 0   0   0  0   0  0
    ## 4 aardsda01   2008     1    BOS   AL 47         5  1 0 0   0   0  0   0  0
    ## 5 aardsda01   2009     1    SEA   AL 73         3  0 0 0   0   0  0   0  0
    ## 6 aardsda01   2010     1    SEA   AL 53         4  0 0 0   0   0  0   0  0
    ##   CS BB SO IBB HBP SH SF GIDP G_old
    ## 1  0  0  0   0   0  0  0    0    11
    ## 2  0  0  0   0   0  1  0    0    45
    ## 3  0  0  0   0   0  0  0    0     2
    ## 4  0  0  1   0   0  0  0    0     5
    ## 5  0  0  0   0   0  0  0    0    NA
    ## 6  0  0  0   0   0  0  0    0    NA

Checking out the type of variables:

    str(batting)

    ## 'data.frame':    97889 obs. of  24 variables:
    ##  $ playerID : Factor w/ 18107 levels "aardsda01","aaronha01",..: 1 1 1 1 1 1 1 2 2 2 ...
    ##  $ yearID   : int  2004 2006 2007 2008 2009 2010 2012 1954 1955 1956 ...
    ##  $ stint    : int  1 1 1 1 1 1 1 1 1 1 ...
    ##  $ teamID   : Factor w/ 149 levels "ALT","ANA","ARI",..: 117 35 33 16 116 116 93 80 80 80 ...
    ##  $ lgID     : Factor w/ 6 levels "AA","AL","FL",..: 4 4 2 2 2 2 2 4 4 4 ...
    ##  $ G        : int  11 45 25 47 73 53 1 122 153 153 ...
    ##  $ G_batting: int  11 43 2 5 3 4 NA 122 153 153 ...
    ##  $ AB       : int  0 2 0 1 0 0 NA 468 602 609 ...
    ##  $ R        : int  0 0 0 0 0 0 NA 58 105 106 ...
    ##  $ H        : int  0 0 0 0 0 0 NA 131 189 200 ...
    ##  $ X2B      : int  0 0 0 0 0 0 NA 27 37 34 ...
    ##  $ X3B      : int  0 0 0 0 0 0 NA 6 9 14 ...
    ##  $ HR       : int  0 0 0 0 0 0 NA 13 27 26 ...
    ##  $ RBI      : int  0 0 0 0 0 0 NA 69 106 92 ...
    ##  $ SB       : int  0 0 0 0 0 0 NA 2 3 2 ...
    ##  $ CS       : int  0 0 0 0 0 0 NA 2 1 4 ...
    ##  $ BB       : int  0 0 0 0 0 0 NA 28 49 37 ...
    ##  $ SO       : int  0 0 0 1 0 0 NA 39 61 54 ...
    ##  $ IBB      : int  0 0 0 0 0 0 NA NA 5 6 ...
    ##  $ HBP      : int  0 0 0 0 0 0 NA 3 3 2 ...
    ##  $ SH       : int  0 1 0 0 0 0 NA 6 7 5 ...
    ##  $ SF       : int  0 0 0 0 0 0 NA 4 4 7 ...
    ##  $ GIDP     : int  0 0 0 0 0 0 NA 13 20 21 ...
    ##  $ G_old    : int  11 45 2 5 NA NA NA 122 153 153 ...

Feature Engineering
-------------------

There are three more statistics that were used in Moneyball, that we
don't have here. So we'll have to calculate them from the data we
currently have.

The statistics are:

-   [Batting Average](https://en.wikipedia.org/wiki/Batting_average)
-   [On Base
    Percentage](http://en.wikipedia.org/wiki/On-base_percentage)
-   [Slugging
    Percentage](http://en.wikipedia.org/wiki/Slugging_percentage)

To begin with, the Batting Average is calculated by dividing the Hits
(H) by At Base(AB).

    batting$BA <- batting$H / batting$AB

    tail(batting$BA,5)

    ## [1] 0.1230769 0.2746479 0.1470588 0.2745098 0.2138728

We can find the formula for On Base Percentage on the wikipedia page
linked above.

    batting <- batting %>%
      mutate(OBP = (H+BB+HBP)/(AB+BB+HBP+SF))

For the Slugging Percentage, we'll need the 1B (singles), which we'll
have to calculate by subtracting doubles, triples and home runs from the
total hits.

Then we'll calculate the slugging percentage using the formula in the
wikipedia article.

    batting <- batting %>%
      mutate(X1B = H - X2B - X3B - HR)

    batting <- batting %>%
      mutate(SLG = (X1B + (2*X2B) + (3*X3B) + (4*HR))/AB)

Merging Batting and Salary Data
-------------------------------

Remember that the Oakland A's had limited revenue, so the players that
we look for have to be cheap, in addition to being good. We loaded two
datasets at the beginning, one for batting statistics and another for
salary data for the players. We'll merge both into a single dataset.

Let's start by looking at the summaries for both datasets.

    summary(batting)

    ##       playerID         yearID         stint           teamID     
    ##  mcguide01:   31   Min.   :1871   Min.   :1.000   CHN    : 4720  
    ##  henderi01:   29   1st Qu.:1931   1st Qu.:1.000   PHI    : 4621  
    ##  newsobo01:   29   Median :1970   Median :1.000   PIT    : 4575  
    ##  johnto01 :   28   Mean   :1962   Mean   :1.077   SLN    : 4535  
    ##  kaatji01 :   28   3rd Qu.:1995   3rd Qu.:1.000   CIN    : 4393  
    ##  ansonca01:   27   Max.   :2013   Max.   :5.000   CLE    : 4318  
    ##  (Other)  :97717                                  (Other):70727  
    ##    lgID             G            G_batting            AB       
    ##  AA  : 1890   Min.   :  1.00   Min.   :  0.00   Min.   :  0.0  
    ##  AL  :44369   1st Qu.: 13.00   1st Qu.:  7.00   1st Qu.:  9.0  
    ##  FL  :  470   Median : 35.00   Median : 32.00   Median : 61.0  
    ##  NL  :49944   Mean   : 51.65   Mean   : 49.13   Mean   :154.1  
    ##  PL  :  147   3rd Qu.: 81.00   3rd Qu.: 81.00   3rd Qu.:260.0  
    ##  UA  :  332   Max.   :165.00   Max.   :165.00   Max.   :716.0  
    ##  NA's:  737                    NA's   :1406     NA's   :6413   
    ##        R                H               X2B            X3B        
    ##  Min.   :  0.00   Min.   :  0.00   Min.   : 0.0   Min.   : 0.000  
    ##  1st Qu.:  0.00   1st Qu.:  1.00   1st Qu.: 0.0   1st Qu.: 0.000  
    ##  Median :  5.00   Median : 12.00   Median : 2.0   Median : 0.000  
    ##  Mean   : 20.47   Mean   : 40.37   Mean   : 6.8   Mean   : 1.424  
    ##  3rd Qu.: 31.00   3rd Qu.: 66.00   3rd Qu.:10.0   3rd Qu.: 2.000  
    ##  Max.   :192.00   Max.   :262.00   Max.   :67.0   Max.   :36.000  
    ##  NA's   :6413     NA's   :6413     NA's   :6413   NA's   :6413    
    ##        HR              RBI               SB                CS        
    ##  Min.   : 0.000   Min.   :  0.00   Min.   :  0.000   Min.   : 0.000  
    ##  1st Qu.: 0.000   1st Qu.:  0.00   1st Qu.:  0.000   1st Qu.: 0.000  
    ##  Median : 0.000   Median :  5.00   Median :  0.000   Median : 0.000  
    ##  Mean   : 3.002   Mean   : 18.47   Mean   :  3.265   Mean   : 1.385  
    ##  3rd Qu.: 3.000   3rd Qu.: 28.00   3rd Qu.:  2.000   3rd Qu.: 1.000  
    ##  Max.   :73.000   Max.   :191.00   Max.   :138.000   Max.   :42.000  
    ##  NA's   :6413     NA's   :6837     NA's   :7713      NA's   :29867   
    ##        BB               SO              IBB              HBP        
    ##  Min.   :  0.00   Min.   :  0.00   Min.   :  0.00   Min.   : 0.000  
    ##  1st Qu.:  0.00   1st Qu.:  2.00   1st Qu.:  0.00   1st Qu.: 0.000  
    ##  Median :  4.00   Median : 11.00   Median :  0.00   Median : 0.000  
    ##  Mean   : 14.21   Mean   : 21.95   Mean   :  1.28   Mean   : 1.136  
    ##  3rd Qu.: 21.00   3rd Qu.: 31.00   3rd Qu.:  1.00   3rd Qu.: 1.000  
    ##  Max.   :232.00   Max.   :223.00   Max.   :120.00   Max.   :51.000  
    ##  NA's   :6413     NA's   :14251    NA's   :42977    NA's   :9233    
    ##        SH               SF             GIDP           G_old       
    ##  Min.   : 0.000   Min.   : 0.0    Min.   : 0.00   Min.   :  0.00  
    ##  1st Qu.: 0.000   1st Qu.: 0.0    1st Qu.: 0.00   1st Qu.: 11.00  
    ##  Median : 1.000   Median : 0.0    Median : 1.00   Median : 34.00  
    ##  Mean   : 2.564   Mean   : 1.2    Mean   : 3.33   Mean   : 50.99  
    ##  3rd Qu.: 3.000   3rd Qu.: 2.0    3rd Qu.: 5.00   3rd Qu.: 82.00  
    ##  Max.   :67.000   Max.   :19.0    Max.   :36.00   Max.   :165.00  
    ##  NA's   :12751    NA's   :42446   NA's   :32521   NA's   :5189    
    ##        BA             OBP             X1B              SLG       
    ##  Min.   :0.000   Min.   :0.00    Min.   :  0.00   Min.   :0.000  
    ##  1st Qu.:0.148   1st Qu.:0.19    1st Qu.:  1.00   1st Qu.:0.179  
    ##  Median :0.231   Median :0.29    Median :  9.00   Median :0.309  
    ##  Mean   :0.209   Mean   :0.26    Mean   : 29.14   Mean   :0.291  
    ##  3rd Qu.:0.275   3rd Qu.:0.34    3rd Qu.: 48.00   3rd Qu.:0.397  
    ##  Max.   :1.000   Max.   :1.00    Max.   :225.00   Max.   :4.000  
    ##  NA's   :13520   NA's   :49115   NA's   :6413     NA's   :13520

    summary(sal)

    ##      yearID         teamID      lgID            playerID    
    ##  Min.   :1985   CLE    :  867   AL:11744   moyerja01:   25  
    ##  1st Qu.:1993   LAN    :  861   NL:12212   vizquom01:   24  
    ##  Median :1999   PHI    :  861              glavito02:   23  
    ##  Mean   :1999   SLN    :  858              bondsba01:   22  
    ##  3rd Qu.:2006   BAL    :  855              griffke02:   22  
    ##  Max.   :2013   NYA    :  855              thomeji01:   22  
    ##                 (Other):18799              (Other)  :23818  
    ##      salary        
    ##  Min.   :       0  
    ##  1st Qu.:  250000  
    ##  Median :  507950  
    ##  Mean   : 1864357  
    ##  3rd Qu.: 2100000  
    ##  Max.   :33000000  
    ## 

The batting data goes back to 1871, whie the salary data starts at 1985.
We'll have to remove batting data before 1985 (which we have no use for
anyways).

    batting <- subset(batting, yearID >= 1985)

    summary(batting)

    ##       playerID         yearID         stint          teamID     
    ##  moyerja01:   27   Min.   :1985   Min.   :1.00   SDN    : 1313  
    ##  mulhote01:   26   1st Qu.:1993   1st Qu.:1.00   CLE    : 1306  
    ##  weathda01:   26   Median :2000   Median :1.00   PIT    : 1299  
    ##  maddugr01:   25   Mean   :2000   Mean   :1.08   NYN    : 1297  
    ##  sierrru01:   25   3rd Qu.:2007   3rd Qu.:1.00   BOS    : 1279  
    ##  thomeji01:   25   Max.   :2013   Max.   :4.00   CIN    : 1279  
    ##  (Other)  :35498                                 (Other):27879  
    ##  lgID             G           G_batting            AB       
    ##  AA:    0   Min.   :  1.0   Min.   :  0.00   Min.   :  0.0  
    ##  AL:17226   1st Qu.: 14.0   1st Qu.:  4.00   1st Qu.:  3.0  
    ##  FL:    0   Median : 34.0   Median : 27.00   Median : 47.0  
    ##  NL:18426   Mean   : 51.7   Mean   : 46.28   Mean   :144.7  
    ##  PL:    0   3rd Qu.: 77.0   3rd Qu.: 77.00   3rd Qu.:241.0  
    ##  UA:    0   Max.   :163.0   Max.   :163.00   Max.   :716.0  
    ##                             NA's   :1406     NA's   :4377   
    ##        R                H               X2B              X3B        
    ##  Min.   :  0.00   Min.   :  0.00   Min.   : 0.000   Min.   : 0.000  
    ##  1st Qu.:  0.00   1st Qu.:  0.00   1st Qu.: 0.000   1st Qu.: 0.000  
    ##  Median :  4.00   Median :  8.00   Median : 1.000   Median : 0.000  
    ##  Mean   : 19.44   Mean   : 37.95   Mean   : 7.293   Mean   : 0.824  
    ##  3rd Qu.: 30.00   3rd Qu.: 61.00   3rd Qu.:11.000   3rd Qu.: 1.000  
    ##  Max.   :152.00   Max.   :262.00   Max.   :59.000   Max.   :23.000  
    ##  NA's   :4377     NA's   :4377     NA's   :4377     NA's   :4377    
    ##        HR              RBI               SB                CS        
    ##  Min.   : 0.000   Min.   :  0.00   Min.   :  0.000   Min.   : 0.000  
    ##  1st Qu.: 0.000   1st Qu.:  0.00   1st Qu.:  0.000   1st Qu.: 0.000  
    ##  Median : 0.000   Median :  3.00   Median :  0.000   Median : 0.000  
    ##  Mean   : 4.169   Mean   : 18.41   Mean   :  2.811   Mean   : 1.219  
    ##  3rd Qu.: 5.000   3rd Qu.: 27.00   3rd Qu.:  2.000   3rd Qu.: 1.000  
    ##  Max.   :73.000   Max.   :165.00   Max.   :110.000   Max.   :29.000  
    ##  NA's   :4377     NA's   :4377     NA's   :4377      NA's   :4377    
    ##        BB               SO              IBB               HBP        
    ##  Min.   :  0.00   Min.   :  0.00   Min.   :  0.000   Min.   : 0.000  
    ##  1st Qu.:  0.00   1st Qu.:  1.00   1st Qu.:  0.000   1st Qu.: 0.000  
    ##  Median :  3.00   Median : 12.00   Median :  0.000   Median : 0.000  
    ##  Mean   : 14.06   Mean   : 27.03   Mean   :  1.171   Mean   : 1.273  
    ##  3rd Qu.: 21.00   3rd Qu.: 42.00   3rd Qu.:  1.000   3rd Qu.: 1.000  
    ##  Max.   :232.00   Max.   :223.00   Max.   :120.000   Max.   :35.000  
    ##  NA's   :4377     NA's   :4377     NA's   :4378      NA's   :4387    
    ##        SH               SF              GIDP           G_old      
    ##  Min.   : 0.000   Min.   : 0.000   Min.   : 0.00   Min.   :  0.0  
    ##  1st Qu.: 0.000   1st Qu.: 0.000   1st Qu.: 0.00   1st Qu.: 11.0  
    ##  Median : 0.000   Median : 0.000   Median : 1.00   Median : 32.0  
    ##  Mean   : 1.465   Mean   : 1.212   Mean   : 3.25   Mean   : 49.7  
    ##  3rd Qu.: 2.000   3rd Qu.: 2.000   3rd Qu.: 5.00   3rd Qu.: 77.0  
    ##  Max.   :39.000   Max.   :17.000   Max.   :35.00   Max.   :163.0  
    ##  NA's   :4377     NA's   :4378     NA's   :4377    NA's   :5189   
    ##        BA             OBP             X1B              SLG       
    ##  Min.   :0.000   Min.   :0.000   Min.   :  0.00   Min.   :0.000  
    ##  1st Qu.:0.136   1st Qu.:0.188   1st Qu.:  0.00   1st Qu.:0.167  
    ##  Median :0.233   Median :0.296   Median :  6.00   Median :0.333  
    ##  Mean   :0.205   Mean   :0.262   Mean   : 25.66   Mean   :0.304  
    ##  3rd Qu.:0.274   3rd Qu.:0.342   3rd Qu.: 42.00   3rd Qu.:0.423  
    ##  Max.   :1.000   Max.   :1.000   Max.   :225.00   Max.   :4.000  
    ##  NA's   :8905    NA's   :8821    NA's   :4377     NA's   :8905

The merge:

    combo <- merge(batting,sal,by=c('playerID','yearID'))

    summary(combo)

    ##       playerID         yearID         stint          teamID.x    
    ##  moyerja01:   27   Min.   :1985   Min.   :1.000   LAN    :  940  
    ##  thomeji01:   25   1st Qu.:1993   1st Qu.:1.000   PHI    :  937  
    ##  weathda01:   25   Median :1999   Median :1.000   BOS    :  935  
    ##  vizquom01:   24   Mean   :1999   Mean   :1.098   NYA    :  928  
    ##  gaettga01:   23   3rd Qu.:2006   3rd Qu.:1.000   CLE    :  920  
    ##  griffke02:   23   Max.   :2013   Max.   :4.000   SDN    :  914  
    ##  (Other)  :25250                                  (Other):19823  
    ##  lgID.x           G            G_batting            AB       
    ##  AA:    0   Min.   :  1.00   Min.   :  0.00   Min.   :  0.0  
    ##  AL:12292   1st Qu.: 26.00   1st Qu.:  8.00   1st Qu.:  5.0  
    ##  FL:    0   Median : 50.00   Median : 42.00   Median : 85.0  
    ##  NL:13105   Mean   : 64.06   Mean   : 57.58   Mean   :182.4  
    ##  PL:    0   3rd Qu.:101.00   3rd Qu.:101.00   3rd Qu.:336.0  
    ##  UA:    0   Max.   :163.00   Max.   :163.00   Max.   :716.0  
    ##                              NA's   :906      NA's   :2661   
    ##        R                H               X2B              X3B        
    ##  Min.   :  0.00   Min.   :  0.00   Min.   : 0.000   Min.   : 0.000  
    ##  1st Qu.:  0.00   1st Qu.:  1.00   1st Qu.: 0.000   1st Qu.: 0.000  
    ##  Median :  9.00   Median : 19.00   Median : 3.000   Median : 0.000  
    ##  Mean   : 24.71   Mean   : 48.18   Mean   : 9.276   Mean   : 1.033  
    ##  3rd Qu.: 43.00   3rd Qu.: 87.25   3rd Qu.:16.000   3rd Qu.: 1.000  
    ##  Max.   :152.00   Max.   :262.00   Max.   :59.000   Max.   :23.000  
    ##  NA's   :2661     NA's   :2661     NA's   :2661     NA's   :2661    
    ##        HR              RBI               SB                CS       
    ##  Min.   : 0.000   Min.   :  0.00   Min.   :  0.000   Min.   : 0.00  
    ##  1st Qu.: 0.000   1st Qu.:  0.00   1st Qu.:  0.000   1st Qu.: 0.00  
    ##  Median : 1.000   Median :  8.00   Median :  0.000   Median : 0.00  
    ##  Mean   : 5.369   Mean   : 23.56   Mean   :  3.568   Mean   : 1.54  
    ##  3rd Qu.: 7.000   3rd Qu.: 39.00   3rd Qu.:  3.000   3rd Qu.: 2.00  
    ##  Max.   :73.000   Max.   :165.00   Max.   :110.000   Max.   :29.00  
    ##  NA's   :2661     NA's   :2661     NA's   :2661      NA's   :2661   
    ##        BB               SO              IBB               HBP        
    ##  Min.   :  0.00   Min.   :  0.00   Min.   :  0.000   Min.   : 0.000  
    ##  1st Qu.:  0.00   1st Qu.:  2.00   1st Qu.:  0.000   1st Qu.: 0.000  
    ##  Median :  6.00   Median : 20.00   Median :  0.000   Median : 0.000  
    ##  Mean   : 17.98   Mean   : 33.52   Mean   :  1.533   Mean   : 1.614  
    ##  3rd Qu.: 29.00   3rd Qu.: 55.00   3rd Qu.:  2.000   3rd Qu.: 2.000  
    ##  Max.   :232.00   Max.   :223.00   Max.   :120.000   Max.   :35.000  
    ##  NA's   :2661     NA's   :2661     NA's   :2662      NA's   :2670    
    ##        SH               SF              GIDP            G_old       
    ##  Min.   : 0.000   Min.   : 0.000   Min.   : 0.000   Min.   :  0.00  
    ##  1st Qu.: 0.000   1st Qu.: 0.000   1st Qu.: 0.000   1st Qu.: 20.00  
    ##  Median : 0.000   Median : 0.000   Median : 2.000   Median : 47.00  
    ##  Mean   : 1.786   Mean   : 1.554   Mean   : 4.127   Mean   : 61.43  
    ##  3rd Qu.: 2.000   3rd Qu.: 2.000   3rd Qu.: 7.000   3rd Qu.:101.00  
    ##  Max.   :39.000   Max.   :17.000   Max.   :35.000   Max.   :163.00  
    ##  NA's   :2661     NA's   :2662     NA's   :2661     NA's   :3414    
    ##        BA             OBP             X1B             SLG       
    ##  Min.   :0.000   Min.   :0.000   Min.   :  0.0   Min.   :0.000  
    ##  1st Qu.:0.160   1st Qu.:0.208   1st Qu.:  0.0   1st Qu.:0.200  
    ##  Median :0.242   Median :0.305   Median : 13.0   Median :0.351  
    ##  Mean   :0.212   Mean   :0.270   Mean   : 32.5   Mean   :0.317  
    ##  3rd Qu.:0.276   3rd Qu.:0.346   3rd Qu.: 59.0   3rd Qu.:0.432  
    ##  Max.   :1.000   Max.   :1.000   Max.   :225.0   Max.   :4.000  
    ##  NA's   :5618    NA's   :5562    NA's   :2661    NA's   :5618   
    ##     teamID.y     lgID.y         salary        
    ##  CLE    :  935   AL:12304   Min.   :       0  
    ##  PIT    :  932   NL:13093   1st Qu.:  255000  
    ##  PHI    :  931              Median :  550000  
    ##  SDN    :  923              Mean   : 1879256  
    ##  LAN    :  921              3rd Qu.: 2150000  
    ##  CIN    :  912              Max.   :33000000  
    ##  (Other):19843

Analyzing the lost players:
---------------------------

We mentioned that Oakland A's lost 3 key players. They were:

-   Jason Giambi (giambja01)
-   Johnny Damon (damonjo01)
-   Rainer Gustavo "Ray" Olmendo (saenzol01)

We'll need to look at their statistics to see what the benchmark is when
we're looking for replacements.

Selecting the lost players:

    lost_players <- subset(combo,playerID %in% c('giambja01','damonjo01','saenzol01'))

2001 was the year when Oakland lost those players. Limiting data to 2001
and required columns:

    lost_players <- lost_players %>%
      subset(yearID==2001) %>%
      select(playerID,H,X2B,X3B,HR,OBP,SLG,BA,AB)

    lost_players

    ##        playerID   H X2B X3B HR       OBP       SLG        BA  AB
    ## 5141  damonjo01 165  34   4  9 0.3235294 0.3633540 0.2562112 644
    ## 7878  giambja01 178  47   2 38 0.4769001 0.6596154 0.3423077 520
    ## 20114 saenzol01  67  21   1  9 0.2911765 0.3836066 0.2196721 305

Limiting source of information to 2001:

    combo_2001 <- subset(combo,yearID==2001)

The constraints for our player search are:

-   The total combined salary of the three players can not exceed 15
    million dollars.
-   Their combined number of At Bats (AB) needs to be equal to or
    greater than the lost players.
-   Their mean OBP had to equal to or greater than the mean OBP of the
    lost players.

<!-- -->

    sum(lost_players$AB)

    ## [1] 1469

    mean(lost_players$OBP)

    ## [1] 0.3638687

So, - Combined AB should be equal to or greater than 1469. - Mean OBP
should be greater than or equal to 0.364

Let's plot the salaries and OBP to get a feel for what we have:

    ggplot(combo_2001,aes(x=OBP,y=salary)) + geom_point()

    ## Warning: Removed 168 rows containing missing values (geom_point).

![png](/public/project-images/moneyball/Plotting%20the%20OBP%20and%20Salary-1.png)

A good threshold for the salary seems to be 8 million, while we
definitely have to look for OBPs above 0.

    combo_2001 <- subset(combo_2001, salary < 8000000 & OBP > 0)

Let's check out the average AB for our players:

    mean(lost_players$AB)

    ## [1] 489.6667

We should be aiming to select players with AB around that mark 489, but
it's okay to go a bit lower than that for our threshold as we'll have
players with higher AB as well.

    combo_2001 <- subset(combo_2001, AB > 450)

Let's arrange our filtered players in the descending order of their AB,
and pick the top 10 as our options.

    options <- head(arrange(combo_2001,desc(OBP)),10)

    select(options, playerID,AB,salary,OBP)

    ##     playerID  AB  salary       OBP
    ## 1  giambja01 520 4103333 0.4769001
    ## 2  heltoto01 587 4950000 0.4316547
    ## 3  berkmla01 577  305000 0.4302326
    ## 4  gonzalu01 609 4833333 0.4285714
    ## 5  martied01 470 5500000 0.4234079
    ## 6  thomeji01 526 7875000 0.4161491
    ## 7  alomaro01 575 7750000 0.4146707
    ## 8  edmonji01 500 6333333 0.4102142
    ## 9  gilesbr02 576 7333333 0.4035608
    ## 10 pujolal01 590  200000 0.4029630

And there we have it, we obviously have to ignore giambja01 (as he's one
of the players we have lost), but we can look at different combinations
of the top players in our list (or explore further down the list).

This finishes the project.
