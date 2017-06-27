---
layout: notebook
title: Inferential Statistics - Do men or women oppose sex education?
skills: Hypothesis testing, R, ggplot, dplyr
external_type: RPubs
external_url: http://rpubs.com/sajal_sharma/inferential_statistics
description: Analysing the GSS (General Social Survey) dataset using R to infer if, in the year 2012, were men, of 18 years or above in the United States, more likely to oppose sex education in public schools than women.
---
---

In this project, we will use inferential statistics to find out if men
or women oppose sex education in public schools. As the data we have is
a sample from the population, it's important to use a hypothesis driven
approach to draw conclusions from the sample.

Let's get started.

*The project was completed as a part of Duke University's 'Introduction
to Probability and Data' online course on Coursera, the first of the
Statistics with R Specialization.*

### Load packages

    library(ggplot2)
    library(dplyr)
    library(statsr)

### Load data

    load("gss.Rdata")

------------------------------------------------------------------------

Part 1: Data
------------

According to the
[documentation](http://gss.norc.org/documents/codebook/GSS_Codebook_intro.pdf),
the data in the General Social Survey (GSS) is collected from an
independently drawn sample of English-speaking persons (and
Spanish-speaking persons, beginning from 2006) 18 years of age or above,
living in non-institutional arrangements within the United States. Full
probability sampling was employed in all surveys beginning from 1977.

As [probability
sampling](http://www.socialresearchmethods.net/kb/sampprob.php) was
used, we know that random selection was was a part of the sampling
process. Thus, the data is generalizable to the population as a whole.
But, since the study is an observational study, and not experimental, we
cannot use it to determine causality between variables in the data.

------------------------------------------------------------------------

Part 2: Research question
-------------------------

In the year 2012, were men, of 18 years or above, more likely to oppose
sex education in public schools than women?

Sex education is an important part of a student's schooling, so they are
aware of their own health and make the correct decisions in life. If we
know that there's a chance of difference between men and women's
attitude towards sex education, we can put in the resources for further
research to determine the cause of this difference, and solve the issue.

------------------------------------------------------------------------

Part 3: Exploratory data analysis
---------------------------------

We'll start with filtering out the data for the year 2012, and then
selecting the variables we're interested in. These are

-   sex: Respondent's sex
-   sexeduc: For or against sex education in public schools


      dataset <- filter(gss, year == 2012) %>%
      select(sex,sexeduc)

Let's look at the type of variables we're dealing with:

    str(dataset)

    ## 'data.frame':    1974 obs. of  2 variables:
    ##  $ sex    : Factor w/ 2 levels "Male","Female": 1 1 1 2 2 2 2 2 2 2 ...
    ##  $ sexeduc: Factor w/ 3 levels "Favor","Oppose",..: 1 NA NA 1 1 1 1 NA 1 NA ...

The sex variable has two levels, and the sexeduc has 3 levels. Let's
plot the data:

    ggplot(dataset, aes(x=sexeduc)) + geom_bar() + ggtitle('Favourability to Sex Education in Public Schools') + xlab('Favour or Oppose Sex Education') + theme_bw()

![png](/public/project-images/gss_inf/unnamed-chunk-3-1.png)

It seems like more people favor sex education and oppose it. Which is
good!

We also have a lot of NA values that we don't want to deal with for our
question, so let's remove those, and take a look at the new
distribution.

    dataset <- na.omit(dataset)

    table(dataset)

    ##         sexeduc
    ## sex      Favor Oppose Depends
    ##   Male     519     64       0
    ##   Female   638     53       0

It looks like that, in our sample, there are more males than females
that oppose sex education. We also have no responses for the 'Depends'
category, so let's remove that and visualize our data.

    dataset <- droplevels(dataset)

    mosaicplot(prop.table(table(dataset),1), main = 'Sex vs Favourability of Sex Education')

![png](/public/project-images/gss_inf/unnamed-chunk-5-1.png)

Now, we'll use statistical inference to know if a larger propertion of
men than women, oppose sex education for the whole population.

------------------------------------------------------------------------

Part 4: Inference
-----------------

#### Hypotheses:

For our question, the null hypotheses is, there's no difference in the
proportions of men and women that oppose sex education.

The alternative hypotheses is, a larger proportion of men, than women,
oppose sex education.

#### Conditions:

We now check the conditions for inference, for a difference in
proportion.

-   Each proportion follows a normal model.
-   The two samples are independent of each other.

And for each model to follow a nearly normal model,

-   The sample observations are independent, which can be assumed true
    as the observations were sampled randomly from a large population.
-   Success-failure condition: We expect to see at-least 10 succeses and
    10 failures in our sample. Even iff we assume that a succes, for our
    question, means opposing sex education, both men and women, have
    more than 10 samples for Favors and Oppositions. So this condition
    holds true as well.

The conditions for inference on a difference in proportion also holds
true because we've determined each proportion to follow a normal model.
And that the two samples are independent from each other because of
random sampling.

#### Methods:

We'll examine the difference in proportions of men and women who oppose
sex education by:

1.  Calculating the p-value for the hypotheses that a higher proportion
    of men oppose sex education, given our sample. Then deciding if the
    p-value is small or large enough to reject, or not reject the
    null hypotheses.

2.  Checking a 95% confidence interval for the difference in the
    proportions, and seeing if it contains 0 i.e. there is
    no difference.

#### P-value:

    inference(x= sex,y = sexeduc, data = dataset, statistic = "proportion", type = "ht", null = 0 ,method = "theoretical",alternative="greater", success = "Oppose")

    ## Response variable: categorical (2 levels, success: Oppose)
    ## Explanatory variable: categorical (2 levels)
    ## n_Male = 583, p_hat_Male = 0.1098
    ## n_Female = 691, p_hat_Female = 0.0767
    ## H0: p_Male =  p_Female
    ## HA: p_Male > p_Female
    ## z = 2.0367
    ## p_value = 0.0208

![png](/public/project-images/gss_inf/unnamed-chunk-6-1.png)

We can see that the p-value is less than 5%, so for this question, we
reject the null hypotheses and conclude that for our population in
general, a larger proportion of males, compared to females, oppose sex
education.

#### Confidence Interval

Reason for including CI: We're constructing a confidence interval for
our hypotheses test, to double check the results of the method with the
p-value test.

    inference(x= sex,y = sexeduc, data = dataset, statistic = "proportion", type = "ci",method = "theoretical", success = "Oppose")

    ## Response variable: categorical (2 levels, success: Oppose)
    ## Explanatory variable: categorical (2 levels)
    ## n_Male = 583, p_hat_Male = 0.1098
    ## n_Female = 691, p_hat_Female = 0.0767
    ## 95% CI (Male - Female): (9e-04 , 0.0653)

![png](/public/project-images/gss_inf/unnamed-chunk-7-1.png)

The 95% confidence interval for the difference between proportions of
males and females that oppose sex education is between 0.0009 to 0.0653.
So even though the difference is really small, it still exists. Hence
the results for both tests agree, and we can say that a greater
proportion of men, than females, oppose sex education, and steps can be
taken to better educate the men on the issue.
