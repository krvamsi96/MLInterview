# Statistics for Data Science
* [Gaussian/Normal](#a)
* [Bernoulli](#b)
* [Central Limit Theorem](#c)
* [Confidence Intervals](#d)
* [Hypothesis Testing](#e)
* [Significance Level](#f)
* [P-value](#g)
* [A/B testing](#h)
* [Correlation &  Multicollinearity?](#i)
* [What is correlation and covariance in statistics?](#l)
* [Statistical Hypothesis Tests :star::star:](#j)
* [What are Loss Function and Cost Functions?  ](#k)
* [How are Entropy and Gini index related?](#m)
* [What is the difference between a univariate analysis and a bivariate analysis?](#n)
* [What is root cause analysis?](#o)
* [What is Chi-Square test ?[(#p)
* [What is skewed Distribution & uniform distribution? ](#q)
* [What is box cox transformation?](#r)
* [VIF](#s)






**Population** : The entire group one desires information about.<br>
**Sample** : A subset of the population taken because the entire population is usually too large to analyze. It's 
characteristics are taken to be representative of the population.</br>
**Mean** : The sum of all the values in the sample divided by the number of values in the sample/population.</br>
**Median** : The median is the value separating the higher half of a data sample from the lower half.</br>
**Standard Deviation** : Square root of the variance. It measures the dispersion around the mean.</br>
**Percentiles** :  An extension of median to values other than 50%. </br> 
**Interquartile range (IQR)** : the difference between the 75th and 25th percentile
**Mode** : The most frequently occuring value
**Range** : Difference between the maximum value and the minimum value.

Notice that most of these fall into one of two categories: they capture either the center of the distribution (e.g., mean, median, mode), or its spread (e.g., variance, IQR, range). These two categories are often called **measures of central tendency** and **measures of dispersion**, respectively.

# Important Distributions <a name="a"></br>
 
**Gaussian/Normal** : The normal distribution is a continuous probability distribution that is symmetrical around its mean, most of the observations cluster around the central peak, and the probabilities for values further away from the mean taper off equally in both directions
  
**Bernoulli** :<a name="b"></br> A Bernoulli random variable can be thought of as the outcome of flipping a biased coin, where the probability of heads is p. To be more precise, a Bernoulli random variable takes on value 1 with probability p and value 0 with probability 1−p. Its expectation is p, and its variance is p(1 − p).

Bernoulli variables are typically used to model binary random variables.
## Central Limit Theorem <a name="c"></br>

The Central Limit Theorem states that the sampling distribution of the sample means approaches a normal distribution as the sample size gets larger — no matter what the shape of the population distribution. This fact holds especially true for sample sizes over 30.

## Confidence Intervals <a name="d"></br>

A confidence interval is how much uncertainty there is with any particular statistic. Confidence intervals are often used with a margin of error. It tells you how confident you can be that the results from a poll or survey reflect what you would expect to find if it were possible to survey the entire population. 

## Hypothesis Testing <a name="e"></br>

Hypothesis testing in statistics is a way for you to test the results of a survey or experiment to see if you have meaningful results. <br>
Hypothesis testing steps:

* Figure out your null hypothesis,
* State your null hypothesis,
* Choose what kind of test you need to perform,
* Either support or reject the null hypothesis.

## Significance Level <a name="f"></br>

The significance level α is the probability of making the wrong decision when the null hypothesis is true. Alpha levels (sometimes just called “significance levels”) are used in hypothesis tests. Usually, these tests are run with an alpha level of .05 (5%), but other levels commonly used are .01 and .10.


## What does it mean when the p-values are high and low? <a name="g"></br>
The p-value is a number between 0 and 1 and interpreted in the following way: A small p-value (typically ≤ 0.05) indicates strong evidence against the null hypothesis, so you reject the null hypothesis.
A large p-value (> 0.05) indicates weak evidence against the null hypothesis, so you fail to reject the null hypothesis. p-values very close to the cutoff (0.05) are considered to be marginal (could go either way).

## A/B testing <a name="h"></br>

A statistical way of comparing two (or more) techniques, typically an incumbent against a new rival. A/B testing aims to determine not only which technique performs better but also to understand whether the difference is statistically significant. A/B testing usually considers only two techniques using one measurement, but it can be applied to any finite number of techniques and measures.

## Correlation <a name="i"></br>

Correlation is the statistical measure of the relationship between two variables.The correlation coefficient, or Pearson's, is calculated using a least-squares measure of the error between an estimating line and the actual data values, normalized by the square root of their variances.

Types of Correlation

* Pearson Correlation Coefficient (measures the linear association between continuous variables)
* Spearman's Correlation (special case of Pearson ρ applied to ranked (sorted) variables. appropriate to use with both continuous and discrete data.) 
* Kendall's Tau (more appropriate for discrete data.)

## What is multicollinearity and how to remove it?
Multicollinearity exists when an independent variable is highly correlated with another independent variable in a multiple regression equation. This can be problematic because it undermines the statistical significance of an independent variable.

You could use the Variance Inflation Factors (VIF) to determine if there is any multicollinearity between independent variables — a standard benchmark is that if the VIF is greater than 5 then multicollinearity exists.



## Statistical Hypothesis Tests :star::star: <a name="j"></br>

 ### Normality Tests (Statistical tests that you can use to check if your data has a Gaussian distribution.)
  * **Shapiro-Wilk Test** : Tests whether a data sample has a Gaussian distribution.</br>
  
 ### Correlation Tests (Statistical tests that you can use to check if two samples are related)
 
  * **Pearson’s Correlation Coefficient** : Tests whether two samples have a monotonic relationship. </br>
   
  * **Spearman’s Rank Correlation** : Tests whether two samples have a monotonic relationship.</br>
   
  * **Chi-Squared Test** : Tests whether two categorical variables are related or independent.</br>
   
## What are Loss Function and Cost Functions?  <a name="k"></br>
the loss function is to capture the difference between the actual and predicted values for a single record whereas 
cost functions aggregate the difference for the entire training dataset.
The Most commonly used loss functions are Mean-squared error and Hinge loss.

## What is correlation and covariance in statistics? <a name="l"></br>
Correlation is defined as the measure of the relationship between two variables. If two variables are directly proportional to each other, then its positive correlation. If the variables are indirectly proportional to each other, it is known as a negative correlation. Covariance is the measure of how much two random variables vary together.

## How are Entropy and Gini index related? <a name="m"></br>
Both Entropy and Gini index are parabolic in nature when it's plotted by the probability. However the maximum value of my Gini Impurity is 0.5 while that of entropy is 1 which only occurs when my classes are Equi-probable. Both of them will tend to zero values when one of the classes are dominant.

## What is the difference between a univariate analysis and a bivariate analysis? <a name="n"></br>
The univariate analysis involves analyzing the distribution of a single variable. Bivariate analysis, on the other hand, considers the relationship between two distinct variables.
For example, calculating the average amount of coffee consumed by a certain population would be a univariate analysis. In contrast, understanding the relationship between coffee consumption and age, gender, or time of the year would be examples of bivariate analysis.

## What is root cause analysis? <a name="o"></br>
Root cause analysis (RCA) is the process of discovering the root causes of problems in order to identify appropriate solutions. It assumes that it is much more effective to systematically prevent and solve for underlying issues rather than just treating ad hoc symptoms and putting out fires.

## What is Chi-Square test ? <a name="p"></br>
A hypothesis testing method is the Chi-square test. Checking if observed frequencies in one or more categories match expected frequencies is one of two frequent Chi-square tests. The Chi-square test is used to determine whether your data is as expected. The test's core premise is to compare the observed values in your data to the expected values that you would see if the null hypothesis is true. The Chi-square goodness of fit test and the Chi-square test of independence are two regularly used Chi-square tests. Both tests use variables to categorize your data into groups.

## What is skewed Distribution & uniform distribution? <a name="q"></br>
Uniform distribution refers to a condition when all the observations in a dataset are equally spread across the range of distribution. Skewed distribution refers to the condition when one side of the graph has more dataset in comparison to the other side.

## What is box cox transformation?  <a name="r"></br>
A Box Cox transformation is a transformation of non-normal dependent variables into a normal shape. Normality is an important assumption for many statistical techniques; if your data isn’t normal, applying a Box-Cox means that you are able to run a broader number of tests.

## VIF <a name="rs"></br>
Variance Inflation Factor (VIF) is used to detect the presence of multicollinearity. Variance inflation factors (VIF) measure how much the variance of the estimated regression coefficients are inflated as compared to when the predictor variables are not linearly related.


















