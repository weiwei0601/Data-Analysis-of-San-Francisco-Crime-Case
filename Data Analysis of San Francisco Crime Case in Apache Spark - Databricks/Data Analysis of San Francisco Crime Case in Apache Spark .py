# Databricks notebook source
# MAGIC %md
# MAGIC # Data Analysis of San Francisco Crime Case in Apache Spark

# COMMAND ----------

# MAGIC %md 
# MAGIC Data source: https://data.sfgov.org/Public-Safety/Police-Department-Incident-Reports-Historical-2003/tmnf-yvry

# COMMAND ----------

# MAGIC %md
# MAGIC Contents
# MAGIC 1. Import Package and Data
# MAGIC 2. OLAP tasks
# MAGIC 3. Conclusions and Suggestions

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Import Package and Data

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.1 Import Package 

# COMMAND ----------

from csv import reader
from pyspark.sql import Row 
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import warnings

import os
os.environ["PYSPARK_PYTHON"] = "python3"

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.2 Import Data

# COMMAND ----------

import urllib.request
urllib.request.urlretrieve("https://data.sfgov.org/api/views/tmnf-yvry/rows.csv?accessType=DOWNLOAD", "/tmp/myxxxx.csv")
dbutils.fs.mv("file:/tmp/myxxxx.csv", "dbfs:/laioffer/spark_hw1/data/sf_03_18.csv")
display(dbutils.fs.ls("dbfs:/laioffer/spark_hw1/data/"))

# COMMAND ----------

data_path = "dbfs:/laioffer/spark_hw1/data/sf_03_18.csv"

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.3 Get dataframe and SQL

# COMMAND ----------

from pyspark.sql import SparkSession
spark = SparkSession \
    .builder \
    .appName("crime analysis") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

df_opt1 = spark.read.format("csv").option("header", "true").load(data_path) 
display(df_opt1)
df_opt1.createOrReplaceTempView("sf_crime")

# COMMAND ----------

type(df_opt1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. OLAP tasks

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1 Counts the number of crimes for different category.

# COMMAND ----------

q1_result = df_opt1.groupBy('category').count().orderBy('count', ascending=False)
display(q1_result)

# COMMAND ----------

#Spark SQL based
crimeCategory = spark.sql("SELECT  category, COUNT(*) AS Count \
                           FROM sf_crime \
                           GROUP BY category \
                           ORDER BY Count DESC")
display(crimeCategory)

# COMMAND ----------

crimes_pd_df = crimeCategory.toPandas()
type(crimes_pd_df)

# COMMAND ----------

display(crimes_pd_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2 Counts the number of crimes for different district, and visualize the results.

# COMMAND ----------

df_q2 = spark.sql("SELECT  pddistrict as district, COUNT(*) AS Count \
                           FROM sf_crime \
                           GROUP BY pddistrict \
                           ORDER BY Count DESC")
display(df_q2)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.3 Count the number of crimes each "Sunday" at "SF downtown".

# COMMAND ----------

# MAGIC %md
# MAGIC I assume SF downtown spacial range: X (-122.4213,-122.4313), Y(37.7540,37.7740).

# COMMAND ----------

df_q3 = spark.sql("select category, count(*) as crime_count \
                   from sf_crime \
                   where DayOfWeek ==  'Sunday' \
                   and X > -122.4313 and X < -122.4213 \
                   and Y > 37.7540 and Y < 37.7740 \
                   group by category \
                   order by crime_count desc")

# COMMAND ----------

display(df_q3)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.4 Analysis the number of crime in each month of 2015, 2016, 2017, 2018.

# COMMAND ----------

from pyspark.sql.functions import *
df_update = df_opt1.withColumn('Date', to_date(col('Date'), 'MM/dd/yyyy'))
display(df_update)

# COMMAND ----------

df_update.createOrReplaceTempView('sf_crime')

# COMMAND ----------

q4_2015 = spark.sql("select * from sf_crime \
                    where year(Date) = '2015' \
                    ")
display(q4_2015)

# COMMAND ----------

q4_2016 = spark.sql("select * from sf_crime \
                    where year(Date) = '2016' \
                    ")

q4_2017 = spark.sql("select * from sf_crime \
                    where year(Date) = '2017' \
                    ")

q4_2018 = spark.sql("select * from sf_crime \
                    where year(Date) = '2018' \
                    ")

# COMMAND ----------

q4_2015.createOrReplaceTempView('sf_crime_2015')
q4_2016.createOrReplaceTempView('sf_crime_2016')
q4_2017.createOrReplaceTempView('sf_crime_2017')
q4_2018.createOrReplaceTempView('sf_crime_2018')

# COMMAND ----------

q4_2015_month = spark.sql("select month(date) as Month, count(*) as number_of_crime \
                          from sf_crime_2015 \
                          group by Month \
                          order by Month asc")
q4_2016_month = spark.sql("select month(date) as Month, count(*) as number_of_crime \
                          from sf_crime_2016 \
                          group by Month \
                          order by Month asc")

q4_2017_month = spark.sql("select month(date) as Month, count(*) as number_of_crime \
                          from sf_crime_2017 \
                          group by Month \
                          order by Month asc")

q4_2018_month = spark.sql("select month(date) as Month, count(*) as number_of_crime \
                          from sf_crime_2018 \
                          group by Month \
                          order by Month asc")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2015

# COMMAND ----------

display(q4_2015_month)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2016

# COMMAND ----------

display(q4_2016_month)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2017

# COMMAND ----------

display(q4_2017_month)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2018

# COMMAND ----------

display(q4_2018_month)

# COMMAND ----------

# MAGIC %md
# MAGIC #### show 2015 to 2018

# COMMAND ----------

q4_2015_month.createOrReplaceTempView('sf_crime_2015_month')
q4_2016_month.createOrReplaceTempView('sf_crime_2016_month')
q4_2017_month.createOrReplaceTempView('sf_crime_2017_month')
q4_2018_month.createOrReplaceTempView('sf_crime_2018_month')

# COMMAND ----------

Q4_union = spark.sql("select Month, number_of_crime from sf_crime_2015_month union select Month, number_of_crime from sf_crime_2016_month union select Month, number_of_crime from sf_crime_2017_month union select Month, number_of_crime from sf_crime_2018_month")

# COMMAND ----------

display(Q4_union)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Comment of Q4: For the first 3 years(2015, 2016 and 2017) the number of crime in each month is relatively stable, 
# MAGIC ##### however, in 2018, it decreases sharply. This may lead to the resurrection of Physical business.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.5 Analysis the number of crime w.r.t the hour in certian day like 2015/12/15, 2016/12/15, 2017/12/15. Then, give your travel suggestion to visit SF. 

# COMMAND ----------

# MAGIC %md
# MAGIC ##### I choose 2015/12/15 to analysis.

# COMMAND ----------

df_q5 = spark.sql("select * from sf_crime \
                   where month(date) == '12' and day(date) == '15' and year(date) == '2015' \
                   ")

# COMMAND ----------

display(df_q5)

# COMMAND ----------

df_q5.createOrReplaceTempView('sf_q5')

# COMMAND ----------

df_q5_hour = spark.sql("select hour(time) as hour, count(*) as number_of_crime from sf_q5 \
                       group by hour \
                       order by hour asc")

# COMMAND ----------

display(df_q5_hour)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Suggestion: According to the plot, we can find that there is risk of crime for every hours, however, the hours in range 10 to 19 have the most frequent rate of crime, should take care when visiting SF at this time range.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.6 Advice to distribute the police
# MAGIC (1) Step1: Find out the top-3 danger disrict  
# MAGIC (2) Step2: find out the crime event w.r.t category and time (hour) from the result of step 1  
# MAGIC (3) give your advice to distribute the police based on your analysis results. 

# COMMAND ----------

# MAGIC %md
# MAGIC #### (1) Step 1

# COMMAND ----------

df_Q6 = spark.sql("select pddistrict as District, count(*) as number_of_crime \
                   from sf_crime \
                   group by District \
                   order by number_of_crime desc")

# COMMAND ----------

display(df_Q6)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### The most dangerous 3 district is 'Southern', 'Mission' and 'Northern'.

# COMMAND ----------

# MAGIC %md
# MAGIC #### (2) Step 2

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Southern

# COMMAND ----------

df_Q6_Southern = spark.sql("select category, hour(time) as Hour, count(*) as number_of_crime \
                            from sf_crime \
                            where pddistrict == 'SOUTHERN' \
                            group by category, Hour \
                            order by Hour asc \
                            ")
display(df_Q6_Southern)

# COMMAND ----------

display(df_Q6_Southern)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Mission

# COMMAND ----------

df_Q6_Mission = spark.sql("select category, hour(time) as Hour, count(*) as number_of_crime \
                            from sf_crime \
                            where pddistrict == 'MISSION' \
                            group by category, Hour \
                            order by Hour asc \
                            ")
display(df_Q6_Mission)

# COMMAND ----------

display(df_Q6_Mission)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Northern

# COMMAND ----------

df_Q6_Northern = spark.sql("select category, hour(time) as Hour, count(*) as number_of_crime \
                            from sf_crime \
                            where pddistrict == 'NORTHERN' \
                            group by category, Hour \
                            order by Hour asc \
                            ")
display(df_Q6_Northern)

# COMMAND ----------

display(df_Q6_Northern)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Advice to distribute the police:
# MAGIC ##### 1. For the 3 district, the most frequent crime type is 'Larceny/Theft'.
# MAGIC ##### 2. For the 3 district, the most of the crimes happen during 14-20 hour.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.7 For different category of crime, find the percentage of them. Based on the output, give my hints to adjust the policy.

# COMMAND ----------

df_Q7_total = spark.sql("select count(*) as total \
                         from sf_crime")
display(df_Q7_total)

# COMMAND ----------

df_Q7 = spark.sql("select category, count(*)/2160953 as percentage \
                  from sf_crime \
                  group by category \
                  order by percentage desc")
display(df_Q7)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Hint 1: Larceny/Theft is the most common crime among all of those crimes.
# MAGIC ##### Hint 2: Nearly half of the crime is contributed by 'Larceny/Theft', 'Other offenses' and 'Non-criminal'.
# MAGIC ##### Police should pay more attention to the categories mentioned above.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.8 For different weekdays, find the percentage of resolution.

# COMMAND ----------

df_Q8 = spark.sql("select DayOfWeek, count(*)/2160953 as percentage \
                   from sf_crime \
                   group by DayOfWeek \
                   order by percentage desc")

# COMMAND ----------

display(df_Q8)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Comment: We can find that the frequency of crime on each weekdays is even and it is slightly higher on Friday, Wednesday and Saturday.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Conclusion and Suggestions

# COMMAND ----------

# MAGIC %md
# MAGIC ##### This project is a kind of data analysis work.
# MAGIC 
# MAGIC ##### The goal of this is to discover whether there are some hidden law or relationship between the crime(amount, type...) and all of the features.(District, time, date...)  
# MAGIC 
# MAGIC ##### In this project, I apply Python spark and SQL to analyze these data via the data structure of Dataframe.
# MAGIC 
# MAGIC ##### To analyze, I also finish some Online Analytical Processing(OLAP) to find the hidden relationship and make some plots to visualize them.
# MAGIC 
# MAGIC ##### Among the results I get, I think the most valuable information can be summarized as the following 4 points:
# MAGIC 
# MAGIC 1. The district 'Southern', 'Mission' and 'Northern' have the highest frequency of crime, the police should pay more attention and strengthen the police at these district.
# MAGIC 
# MAGIC 2. 'Larceny/Theft' is the most frequent type of crime, the police should pay more attention and tell the residents to pay attention to this type of crime.
# MAGIC 
# MAGIC 3. 14-20 is the time range that have the most frequent crime happening, the police should strengthen the police at that time range and tell the residents to pay more attention at that time range.
# MAGIC 
# MAGIC 4. Roughly and on the whole, the frequency of crime is decreasing from 2018, however, that may because of the data missing on this year, more data for this year is needed.

# COMMAND ----------


