---
layout: notebook
title: Behavioral Risk Factor Surveillance System 2013 Exploratory Data Analysis
skills: Descriptive Statistics, R, ggplot, dplyr
external_type: RPubs
external_url: http://rpubs.com/sajal_sharma/brfss2013
description: Analysis of the BRFSS-2013 data set using R, focusing on investigating the relationship between education and eating habits, sleep and mental health, and smoking, drinking and general health of a person.
---
---

In this project, we carry out exploratory analysis of the BRFSS-2013
data set by setting out research questions, and then exploring
relationship between identified variables to answer those questions. To
know more about BRFSS and the dataset, visit [this
link](http://www.cdc.gov/brfss/).

*The project was completed as a part of Duke University's 'Introduction
to Probability and Data' online course on Coursera, the first of the
Statistics with R Specialization.*

Setup
-----

**Load packages**

    library(ggplot2)
    library(dplyr)

**Load data**

    load("brfss2013.RData")

------------------------------------------------------------------------

The Data
--------

The BRFSS-2013 dataset was sampled from the non-institutionalised adult
population (i.e. 18 years and older) residing in the US. The data was
collected through landline and cellular-telephone based surveys.

Disproportionate stratified sampling, which is more efficient than
simple random sampling, was used for the landline sample
([source](http://www.cdc.gov/brfss/data_documentation/pdf/userguidejune2013.pdf)).
The cellular sample was generated from randomly selected respondents,
with an equal probability of selection.

As random sampling was used for both data collection methods, the data
for the sample is generalizable to the population. On the other hand, as
this is an observational study, it won't be possible to make causal
inferences from the data.

------------------------------------------------------------------------

Research questions
------------------

**Research question 1:**

Are non-smoking heavy drinkers, generally healthier than regular
smokers, who are not heavy drinkers?

While researching this, we're trying to explore the impact of consuming
alcohol vs smoking tobacco on a person's health and see which is worse.

**Research question 2:**

Do people who sleep fewer hours than average person, also have more than
days with poor mental health?

Research has suggested that inadequate sleep has a negative effect on a
person's overall health. Here we try to determine if it also has a
negative effect on their mental health.

**Research question 3:**

Are people who have completed higher levels of education, more likely to
consume fruits and vegetables once or more in a day?

We might assume that educated people live a healthier lifestyle i.e.
exercising or eating nutritious food. We'll try and figure out if that's
the case here by comparing education levels with fruit and vegetable
consumption.

------------------------------------------------------------------------

Exploratory data analysis
-------------------------

### Research question 1:

Are non-smoking heavy drinkers, generally healthier than regular
smokers, who are not heavy drinkers?

We'll be using the following variables for this question:

-   genhlth: Respondent's health, in general
-   \_rfsmok3: Is the respondent a current smoker?
-   \_rfdrhv4: Is the respondent a heavy drinker?

Type of the variables we're dealing with:

    str(select(brfss2013,genhlth,X_rfsmok3,X_rfdrhv4))

    ## 'data.frame':    491775 obs. of  3 variables:
    ##  $ genhlth  : Factor w/ 5 levels "Excellent","Very good",..: 4 3 3 2 3 2 4 3 1 3 ...
    ##  $ X_rfsmok3: Factor w/ 2 levels "No","Yes": 1 1 2 1 1 1 1 2 1 1 ...
    ##  $ X_rfdrhv4: Factor w/ 2 levels "No","Yes": 1 1 2 1 1 1 1 1 1 1 ...

All of the above are categorical variable. General health of a person is
defined in 5 levels, while a person is or isn't a heavy drinker or a
smoker.

To begin, let's check out our selected variables individually.

**genhlth: General Health**

    total_obs <- nrow(brfss2013)


    brfss2013 %>%
      group_by(genhlth) %>%
      summarise(count=n(),percentage=n()*100/total_obs)

    ## # A tibble: 6 × 3
    ##     genhlth  count percentage
    ##      <fctr>  <int>      <dbl>
    ## 1 Excellent  85482 17.3823395
    ## 2 Very good 159076 32.3473133
    ## 3      Good 150555 30.6146103
    ## 4      Fair  66726 13.5684002
    ## 5      Poor  27951  5.6836968
    ## 6        NA   1985  0.4036399

    ggplot(brfss2013, aes(x=genhlth)) + geom_bar() + ggtitle('General Health of Respondents') + xlab('General Health') + theme_bw()

![png](/public/project-images/brfss2013/unnamed-chunk-3-1.png)

Around 80% of the respondents in our dataset are in good health or
better, and most of the people have 'Very good' health. There are some
missing (NA) values too which we'll deal with later as they don't make
much sense with our analysis.

**\_rfsmok3: Currently a smoker?**

According to the codebook, respondents who have replied 'Yes', now smoke
every day or some days; while those who replied 'No' have either never
smoked in their lifetimes or don't smoke now.

    brfss2013 %>%
      group_by(X_rfsmok3) %>%
      summarise(count=n(),percentage=n()*100/total_obs)

    ## # A tibble: 3 × 3
    ##   X_rfsmok3  count percentage
    ##      <fctr>  <int>      <dbl>
    ## 1        No 399786  81.294494
    ## 2       Yes  76654  15.587210
    ## 3        NA  15335   3.118296

    ggplot(brfss2013, aes(x=X_rfsmok3)) + geom_bar() + ggtitle('Smoking Status of Respondents') + xlab('Currently a smoker?')+ theme_bw()

![png](/public/project-images/brfss2013/unnamed-chunk-5-1.png)

More than 81% of the respondents are not current smokers, though they
might have smoked earlier in their lifetimes.

**\_rfdrhv4: Heavy drinker?**

The heavy drinker variable is defined as *adult men having more than two
drinks per day and adult women having more than one drink per day)*.

    brfss2013 %>%
      group_by(X_rfdrhv4) %>%
      summarise(count=n(),percentage=n()*100/total_obs)

    ## # A tibble: 3 × 3
    ##   X_rfdrhv4  count percentage
    ##      <fctr>  <int>      <dbl>
    ## 1        No 442359  89.951502
    ## 2       Yes  25533   5.192009
    ## 3        NA  23883   4.856489

    ggplot(brfss2013, aes(x=X_rfdrhv4)) + geom_bar() + ggtitle('Drinking Habits of Respondents') + xlab('Heavy Drinker?') +theme_bw()

![png](/public/project-images/brfss2013/unnamed-chunk-7-1.png)

Only about 5% of the respondends in our dataset are heavy drinkers.

Now to answer our original question, we can make things a bit easier for
ourselves by creating a new categorical variable to categorise a person
as: 'Smoker', 'Heavy Drinker', 'Both' or 'None.'

    brfss2013 <- brfss2013 %>%
      mutate(smoke_alc = ifelse(X_rfdrhv4 == 'Yes',
                                ifelse(X_rfsmok3 == 'Yes','Both','Heavy Drinker'),
                                ifelse(X_rfsmok3 == 'Yes','Current Smoker','None')))

Let's check out the distribution of our new variable:

    brfss2013 %>%
      group_by(smoke_alc) %>%
      summarise(count=n(),percentage=n()*100/total_obs)

    ## # A tibble: 5 × 3
    ##        smoke_alc  count percentage
    ##            <chr>  <int>      <dbl>
    ## 1           Both   8144   1.656042
    ## 2 Current Smoker  66000  13.420772
    ## 3  Heavy Drinker  17269   3.511565
    ## 4           None 374377  76.127701
    ## 5           <NA>  25985   5.283920

    ggplot(brfss2013,aes(x=smoke_alc)) + geom_bar() + ggtitle('Drinking and Smoking Habits of Respondents') + xlab('Drinker or Smoker?') +theme_bw()

![png](/public/project-images/brfss2013/unnamed-chunk-10-1.png)

About 76% of the respondents don't smoke or drink heavily. Around 13.4%
are current smokers, and about 3.5% drink heavily. We'll be focusing on
the last two.

A good way to represent the counts of two categorical variables is a
contingency table.

    rq1_table <- table(brfss2013$smoke_alc,brfss2013$genhlth)

    rq1_table

    ##                 
    ##                  Excellent Very good   Good   Fair   Poor
    ##   Both                 998      2428   2957   1273    446
    ##   Current Smoker      6637     17160  22372  12828   6741
    ##   Heavy Drinker       4140      6729   4648   1356    349
    ##   None               69056    125370 112123  47522  18884

It's a little hard to look at the number and quickly understand what
proportions of Current Smokers or Heavy Drinkers have the better health.
So we'll calculate those proportions, and make sure that the rows sum to
1. So we're calculating the proportions of health across drinker or
smokers.

    prop.table(rq1_table,1)

    ##                 
    ##                   Excellent  Very good       Good       Fair       Poor
    ##   Both           0.12317946 0.29967909 0.36497161 0.15712170 0.05504814
    ##   Current Smoker 0.10096139 0.26103623 0.34032067 0.19513828 0.10254343
    ##   Heavy Drinker  0.24039020 0.39072117 0.26988735 0.07873650 0.02026478
    ##   None           0.18515907 0.33615316 0.30063412 0.12742020 0.05063345

Now we have a sense of what's going on. Let's visualize the table
through a mosaic plot.

    mosaicplot(prop.table(rq1_table,1),main='Drinking and/or Smoking vs General Health', xlab='Drinking and/or Smoking status', ylab='General Health')

![png](/public/project-images/brfss2013/unnamed-chunk-13-1.png)

Looking at the summary statistics and the visualization, we can see
that, compared to the 'Current Smoker' category, there is a higher
proportion of 'Heavy Drinkers' with 'Excellent' or 'Very good' health.
Even though there are proportionally more smokers with 'Good' health,
heavy drinkers have lower 'Fair' or 'Poor' health, something we can
consider below par.

Hence, it looks like smokers have poorer health than heavy drinkers.

### Research question 2:

Do people who sleep fewer hours than average person, also have more than
days with poor mental health?

For this, we have to look at the relationship between the variables:

-   sleptim1: On average, the hours of sleep a person gets in a
    24-hour period.
-   menthlth: Out of 30, number of days the mental health of a person
    wasn't good.

Checking out the type of variab;es that we're dealing with:

    str(select(brfss2013,sleptim1,menthlth))

    ## 'data.frame':    491775 obs. of  2 variables:
    ##  $ sleptim1: int  NA 6 9 8 6 8 7 6 8 8 ...
    ##  $ menthlth: int  29 0 2 0 2 0 15 0 0 0 ...

Both of the above variables are continuous integers, but we can also
think of time slept as a categorical variable, and then calculate the
average number of days with poor mental health for a person who gets
that much amount of sleep, to answer our question. We'll do that in a
bit.

**sleptim1**

Taking a look at how the *sleptim1* variable is distributed.

    ggplot(brfss2013,aes(x=sleptim1)) + geom_bar()

    ## Warning: Removed 7387 rows containing non-finite values (stat_count).

![png](/public/project-images/brfss2013/unnamed-chunk-15-1.png)

The plot extends to more than 400 on the x-axis, which is a bit
suspicious. We'll have to check for unrealistic values for *sleptim1* in
our dataset. Let's filter for observations where the time slept is above
the 24 hour period.

    brfss2013 %>%
      filter(sleptim1>24) %>%
      select(sleptim1)

    ##   sleptim1
    ## 1      103
    ## 2      450

As suspected, there are two unrealistic values there, and we'll have to
filter for them when doing our analyis or visualisation. One thing we
can do to avoid repeated filters for this section is to make a new
*clean* dataframe.

    rq2_brfss2013 <- brfss2013 %>%
      filter(sleptim1 <= 24) 

Also, since there are only 24 (25 if you count 0) possible values for
the amount of time slept, we can consider this variable as a factor.
Attempting the plot again:

    ggplot(rq2_brfss2013,aes(x=as.factor(sleptim1))) + geom_bar() + ggtitle('Amount of Sleep of Respondents') + xlab('Hours slept') + theme_bw()

![png](/public/project-images/brfss2013/unnamed-chunk-18-1.png)

Better. It looks like most people get 6-8 hours of sleep. What is the
average hours of sleep for our data?

    rq2_brfss2013 %>%
      summarise(avg_sleep = mean(sleptim1))

    ##   avg_sleep
    ## 1  7.050986

So we can consider 7 hrs to be the optimum/average amount of sleep for
our population.

**menthlth**

    ggplot(rq2_brfss2013, aes(x=menthlth)) + geom_histogram() 

    ## `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.

    ## Warning: Removed 7771 rows containing non-finite values (stat_bin).

![png](/public/project-images/brfss2013/unnamed-chunk-20-1.png)

Looks like we might have the same problem here as with the previous
variable. Removing impossible outliers:

    rq2_brfss2013 <- rq2_brfss2013 %>%
      filter(menthlth <= 30)

    ggplot(rq2_brfss2013, aes(x=menthlth)) + geom_histogram() + ggtitle('Mental Health of Respondents') + xlab('Number of days with poor mental health (out of 30)') + theme_bw()

    ## `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.

![png](/public/project-images/brfss2013/unnamed-chunk-22-1.png)

To answer our question, we can look at the average mental health of
groups of people who get different hours of sleep.

    slep_ment <- rq2_brfss2013 %>%
     group_by(hours_slept = as.factor(sleptim1)) %>%
     summarise(avg_poor_mental = mean(menthlth), count=n()) 

    slep_ment

    ## # A tibble: 24 × 3
    ##    hours_slept avg_poor_mental  count
    ##         <fctr>           <dbl>  <int>
    ## 1            1       11.651376    218
    ## 2            2       13.286550   1026
    ## 3            3       12.265064   3369
    ## 4            4        9.772228  13834
    ## 5            5        6.280284  32699
    ## 6            6        3.933364 104493
    ## 7            7        2.206032 140794
    ## 8            8        2.217405 138934
    ## 9            9        2.739771  23414
    ## 10          10        4.686248  11831
    ## # ... with 14 more rows

Visualizing the data:

    ggplot(slep_ment, aes(x=hours_slept,y=avg_poor_mental)) + geom_bar(stat='identity') + ggtitle('Do people who sleep irregularly have poor mental health?') + xlab('Hours Slept') + ylab('No. of days with poor mental health (out of 30)') + theme_bw()

![png](/public/project-images/brfss2013/unnamed-chunk-24-1.png)

Looking at the summary statistics and the bar graph, we can see that
people who get around 6-9 hours of sleep per day have considerably lower
number of days with poor mental health, though it's even better if you
get 7 or 8 hours of sleep.

People with lesser, or even more than average hours of sleep have more
days of poor mental health. The observation with 23 hours of sleep and
30 days of poor mental health is caused by outliers with scarce data, as
are others, possibly.

We can check this as well:

    slep_ment %>%
      filter(as.integer(hours_slept) > 12)

    ## # A tibble: 12 × 3
    ##    hours_slept avg_poor_mental count
    ##         <fctr>           <dbl> <int>
    ## 1           13        9.448454   194
    ## 2           14        8.567198   439
    ## 3           15        8.424855   346
    ## 4           16        6.719547   353
    ## 5           17        8.428571    35
    ## 6           18       10.318750   160
    ## 7           19       10.000000    12
    ## 8           20        8.548387    62
    ## 9           21        7.500000     2
    ## 10          22        8.200000    10
    ## 11          23       30.000000     3
    ## 12          24        7.218750    32

So it does seem that there is a relationship between inadequate sleep
and mental health, though we cannot be absolutely sure if one directly
causes the other. But, people who sleep lower or more than average, are
also those who suffer from poorer mental health than those who sleep
adequately.

### Research question 3:

Are people who have completed higher levels of education, more likely to
consume fruits and vegetables once or mor in a day?

For this we'll be using the following variables:

-   \_educag: Computed level of education completed.
-   \_frtlt1: Consume fruit 1 or times per day.
-   \_veglt1: Consume vegetables 1 or times per day.

<!-- -->

    str(select(brfss2013,X_educag,X_frtlt1,X_veglt1))

    ## 'data.frame':    491775 obs. of  3 variables:
    ##  $ X_educag: Factor w/ 4 levels "Did not graduate high school",..: 4 3 4 2 4 4 2 3 4 2 ...
    ##  $ X_frtlt1: Factor w/ 2 levels "Consumed fruit one or more times per day",..: 1 2 2 2 2 1 1 2 1 2 ...
    ##  $ X_veglt1: Factor w/ 2 levels "Consumed vegetables one or more times per day",..: 2 1 1 1 1 1 1 1 1 1 ...

All three of these are categorical variables, so we can answer our
question through a contingency table.

Before proceeding, we'll change the name of the factor levels, as they
make the outplot and visualizations look better. For this purpose, we;ll
assume that college and technical school the same thing in \_educag
variable.

    levels(brfss2013$X_frtlt1) <- c('Once or more a day','Less than once a day')
    levels(brfss2013$X_veglt1) <- c('Once or more a day','Less than once a day')

    levels(brfss2013$X_educag)[3] <- c('Attended college')
    levels(brfss2013$X_educag)[4] <- c('Graduated college')

Beginning with an examination of the variables on their own.

**\_educag**

    brfss2013 %>%
      group_by(X_educag) %>%
      summarise(count=n(), percentage=n()*100/total_obs)

    ## # A tibble: 5 × 3
    ##                       X_educag  count percentage
    ##                         <fctr>  <int>      <dbl>
    ## 1 Did not graduate high school  42213  8.5838036
    ## 2        Graduated high school 142968 29.0718316
    ## 3             Attended college 134196 27.2880891
    ## 4            Graduated college 170118 34.5926491
    ## 5                           NA   2280  0.4636267

    ggplot(brfss2013, aes(x=X_educag)) + geom_bar() + ggtitle('Education Level of Respondents') + xlab('Completed Education Level') + theme_bw() + theme(axis.text.x = element_text(angle = 30, hjust = 1))

![png](/public/project-images/brfss2013/unnamed-chunk-29-1.png)

Around 90% of the respondents have graduated high school or higher,
while 8.5% did not graduate high school. In the context of our question,
we expect that as we go up the completed level of education, we'll see
higher proportions of people who consume fruits and vegetables. We'll
come back to that later.

**\_frtlt1**

    brfss2013 %>%
      group_by(X_frtlt1) %>%
      summarise(count=n(), percentage=n()*100/total_obs)

    ## # A tibble: 3 × 3
    ##               X_frtlt1  count percentage
    ##                 <fctr>  <int>      <dbl>
    ## 1   Once or more a day 291729  59.321641
    ## 2 Less than once a day 171343  34.841747
    ## 3                   NA  28703   5.836612

    ggplot(brfss2013, aes(x=X_frtlt1)) + geom_bar() + ggtitle('Fruit Consumption by Respondents') + xlab('Fruit Consumption') + theme_bw() + theme(axis.text.x = element_text(angle = 30, hjust = 1))

![png](/public/project-images/brfss2013/unnamed-chunk-31-1.png)

More people in our dataset, around 59% consume fruits one or more times
per day than those who don't, around 34.8%.

**\_veglt1**

    brfss2013 %>%
      group_by(X_veglt1) %>%
      summarise(count=n(), percentage=n()*100/total_obs)

    ## # A tibble: 3 × 3
    ##               X_veglt1  count percentage
    ##                 <fctr>  <int>      <dbl>
    ## 1   Once or more a day 359834  73.170454
    ## 2 Less than once a day 101777  20.695847
    ## 3                   NA  30164   6.133699

    ggplot(brfss2013, aes(x=X_veglt1)) + geom_bar() + ggtitle('Vegetable Consumption by Respondents') + xlab('Vegetable Consumption') + theme_bw() + theme(axis.text.x = element_text(angle = 30, hjust = 1))

![png](/public/project-images/brfss2013/unnamed-chunk-33-1.png)

Around 73% people in our dataset consume vegetables one or more times
per day, while around 20.7% don't. We can also see that there are more
people that consume veggies (one or more times per day), than there are
who consume fruits.

Answering our question, we'll build a 2-way contingency table to count
frequencies of completed education level with both fruit and vegetable
consumption.

    ct_f <- table(brfss2013$X_educag,brfss2013$X_frtlt1)

    prop.table(ct_f,1)

    ##                               
    ##                                Once or more a day Less than once a day
    ##   Did not graduate high school          0.5352677            0.4647323
    ##   Graduated high school                 0.5734143            0.4265857
    ##   Attended college                      0.6220397            0.3779603
    ##   Graduated college                     0.7052274            0.2947726

    mosaicplot(prop.table(ct_f,1), main='Completed Education vs Fruit Consumption', xlab='Completed Education Level', ylab='Fruit Consumption')

![png](/public/project-images/brfss2013/unnamed-chunk-35-1.png)

In both the proportional frequency table, and the mosaic, we can see an
increase in the proportion of people who consume fruits, as we move to
increased levels of completed education, vs those who don't. Let's see
if this also holds true for vegetable consumption.

    ct_v <- table(brfss2013$X_educag,brfss2013$X_veglt1)

    prop.table(ct_v,1)

    ##                               
    ##                                Once or more a day Less than once a day
    ##   Did not graduate high school          0.6435618            0.3564382
    ##   Graduated high school                 0.7123442            0.2876558
    ##   Attended college                      0.7909306            0.2090694
    ##   Graduated college                     0.8585728            0.1414272

    mosaicplot(prop.table(ct_v,1), main='Completed Education vs Vegetable Consumption', xlab='Completed Education Level', ylab='Vegetable Consumption')

![png](/public/project-images/brfss2013/unnamed-chunk-37-1.png)

There was already a higher proportion of people who consumed vegetables
than those who consumed fruits, but the increase in proportions,
depending on completed education level is still evident here.

So, we can say that people with higher education levels are more likely
to adopt healthy eating habits.
