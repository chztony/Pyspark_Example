import pandas as pd
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.conf import SparkConf
from pyspark.sql import HiveContext
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql.functions import *
from pyspark.sql.types import *


sc = SparkContext()
hive_context = HiveContext(sc)

# Read data from data frame
df_test = hive_context.sql("""
    select *
    from table_1 t1 inner join table_2 t2
    on t1.id = t2.id
 """)


# Define a function to extract the id from the request URLs
def extract_product_id(url_input):
    url = str(url_input)
    if 'id=' in url:
        if '&' in url:
            start_pos = url.find('id=') + 1
            last_pos = url.find('&')
            return url[start_pos:last_pos]
        else:
            start_pos = url.find('id=') + 1
            return url[start_pos::]
    else:
        return None



udf_product_id = udf(extract_product_id, StringType())
df_test_with_id = df_test_with_id.withColumn("new_id", udf_product_id("requesturl"))


# Continue Data Frame Operations
sqlContext.registerDataFrameAsTable(df_test_with_id, "df_test_with_id")

df_test_sub_sql = sqlContext.sql("""
    select *, "add new columns or other operations"
    from df_test_with_id
""")

# Cache the data before actual collect
df_test_sub_sql.cache()
df_test_sub_sql.count()

# Because the data set is cached, the data frame will show almost immediately
df_test_sub_sql.show()

# Clear cache to release context
sqlContext.clearCache()

# Change the data set to Pandas
pd_result = df_test_sub.toPandas()

# Continue Pandas Operations

pd_result.head()

# Convert the pandas data frame back to Spark

df_result = sqlContext.createDataFrame(pd_result)

# Write the data frame back to Hive
df_result.write.mode("overwrite").saveAsTable("analytics.df_result")

# Release the memory
sqlContext.clearCache()

# Most importantly, release spark context
sc.stop()


# Another example of Spark Machine Learning
from pyspark.ml.evaluation import *
from pyspark.mllib.recommendation import *

# Difference between ML and MLLIB:
# https://stackoverflow.com/questions/38835829/whats-the-difference-between-spark-ml-and-mllib-packages

df_result = df_result.select('client_id', 'product_id', 'frequency')

# Transfer the columns into right data type
df_result = df_result.withColumn('client_id', df_result['client_id'].cast(StringType()))
df_result = df_result.withColumn('product_id', df_result['product_id'].cast(StringType()))

# Use MLLIB to train
trainingRDD = df_result.rdd
model = ALS.trainImplicit(trainingRDD, 20, 20)

n = 5
df_prediction = model.recommendProductsForUsers(n)



# Gred Search Process:
train, test = df_result.randomSplit([0.8, 0.2])
trainingRDD = train.rdd
trainingRDD.cache()
x_test = test.drop("frequency")
testingRDD = x_test.rdd
testingRDD.cache()


# Set grid-search parameters
parameters = [(10, 10), (10, 20), (10, 5), (10, 20), (20, 5), (20, 20)]

# Set a performance list to store the result
performance_list = []

# Start grid-search, and it will take a long time to run through all parameter combinations
for parameter in parameters:
    rank = parameter[0]
    iteration = parameter[1]
    print("Rank:", rank, "Iteration:", iteration)
    # Use training set to build the engine
    model_test = ALS.trainImplicit(trainingRDD, rank, iteration)
    # Use testing set to get a prediction
    df_predict = model_test.predictAll(testingRDD).toDF()
    # Join back the actual ratings to the predictions
    df_predict = df_predict.join(test, ((df_predict.user == test.user_id) & (df_predict.product == test.product_id)))
    # Use RMSE evaluator to check the performance. The lower the rmse, the better is the performance
    evaluator = RegressionEvaluator(metricName='rmse', labelCol='usage_count', predictionCol='rating')
    rmse = evaluator.evaluate(df_predict)
    # Store the result into the result list
    performance_list.append((rmse, rank, iteration))
    print("MODEL_RMSE:", rmse, "Rank:", rank, "Iteration:", iteration)