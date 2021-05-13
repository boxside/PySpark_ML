# Logistic Regression Consulting Project

## Binary Customer Churn

A marketing agency has many customers that use their service to produce ads for the client/customer websites. They've noticed that they have quite a bit of churn in clients. They basically randomly assign account managers right now, but want you to create a machine learning model that will help predict which customers will churn (stop buying their service) so that they can correctly assign the customers most at risk to churn an account manager. Luckily they have some historical data, can you help them out? Create a classification algorithm that will help classify whether or not a customer churned. Then the company can test this against incoming data for future customers to predict which customers will churn and assign them an account manager.

The data is saved as customer_churn.csv. Here are the fields and their definitions:

    Name : Name of the latest contact at Company
    Age: Customer Age
    Total_Purchase: Total Ads Purchased
    Account_Manager: Binary 0=No manager, 1= Account manager assigned
    Years: Totaly Years as a customer
    Num_sites: Number of websites that use the service.
    Onboard_date: Date that the name of the latest contact was onboarded
    Location: Client HQ Address
    Company: Name of Client Company
    
Once you've created the model and evaluated it, test out the model on some new data (you can think of this almost like a hold-out set) that your client has provided, saved under new_customers.csv. The client wants to know which customers are most likely to churn given this data (they don't have the label yet).


```python
from pyspark.sql import SparkSession
```


```python
spark = SparkSession.builder.appName('logregconsult').getOrCreate()
```


```python
data = spark.read.csv('customer_churn.csv',inferSchema=True,
                     header=True)
```


```python
data.printSchema()
```

    root
     |-- Names: string (nullable = true)
     |-- Age: double (nullable = true)
     |-- Total_Purchase: double (nullable = true)
     |-- Account_Manager: integer (nullable = true)
     |-- Years: double (nullable = true)
     |-- Num_Sites: double (nullable = true)
     |-- Onboard_date: timestamp (nullable = true)
     |-- Location: string (nullable = true)
     |-- Company: string (nullable = true)
     |-- Churn: integer (nullable = true)
    
    

### Check out the data


```python
data.describe().show()
```

    +-------+-----------------+-----------------+------------------+-----------------+------------------+-------------------+
    |summary|              Age|   Total_Purchase|   Account_Manager|            Years|         Num_Sites|              Churn|
    +-------+-----------------+-----------------+------------------+-----------------+------------------+-------------------+
    |  count|              900|              900|               900|              900|               900|                900|
    |   mean|41.81666666666667|10062.82403333334|0.4811111111111111| 5.27315555555555| 8.587777777777777|0.16666666666666666|
    | stddev|6.127560416916251|2408.644531858096|0.4999208935073339|1.274449013194616|1.7648355920350969| 0.3728852122772358|
    |    min|             22.0|            100.0|                 0|              1.0|               3.0|                  0|
    |    max|             65.0|         18026.01|                 1|             9.15|              14.0|                  1|
    +-------+-----------------+-----------------+------------------+-----------------+------------------+-------------------+
    
    


```python
data.columns
```




    ['Names',
     'Age',
     'Total_Purchase',
     'Account_Manager',
     'Years',
     'Num_Sites',
     'Onboard_date',
     'Location',
     'Company',
     'Churn']



### Format for MLlib

We'll ues the numerical columns. We'll include Account Manager because its easy enough, but keep in mind it probably won't be any sort of a signal because the agency mentioned its randomly assigned!


```python
from pyspark.ml.feature import VectorAssembler
```


```python
assembler = VectorAssembler(inputCols=['Age',
 'Total_Purchase',
 'Account_Manager',
 'Years',
 'Num_Sites'],outputCol='features')
```


```python
output = assembler.transform(data)
```


```python
final_data = output.select('features','churn')
```

### Test Train Split


```python
train_churn,test_churn = final_data.randomSplit([0.7,0.3])
```

### Fit the model


```python
from pyspark.ml.classification import LogisticRegression
```


```python
lr_churn = LogisticRegression(labelCol='churn')
```


```python
fitted_churn_model = lr_churn.fit(train_churn)
```


```python
training_sum = fitted_churn_model.summary
```


```python
training_sum.predictions.describe().show()
```

    +-------+-------------------+-------------------+
    |summary|              churn|         prediction|
    +-------+-------------------+-------------------+
    |  count|                632|                632|
    |   mean|0.16772151898734178|0.13924050632911392|
    | stddev|0.37391474020622584| 0.3464715405857694|
    |    min|                  0|                0.0|
    |    max|                  1|                1.0|
    +-------+-------------------+-------------------+
    
    

### Evaluate results

Let's evaluate the results on the data set we were given (using the test data)


```python
from pyspark.ml.evaluation import BinaryClassificationEvaluator
```


```python
pred_and_labels = fitted_churn_model.evaluate(test_churn)
```


```python
pred_and_labels.predictions.show()
```

    +--------------------+-----+--------------------+--------------------+----------+
    |            features|churn|       rawPrediction|         probability|prediction|
    +--------------------+-----+--------------------+--------------------+----------+
    |[29.0,11274.46,1....|    0|[4.87277048314045...|[0.99240597473215...|       0.0|
    |[30.0,8403.78,1.0...|    0|[6.62706699787450...|[0.99867770995491...|       0.0|
    |[30.0,8874.83,0.0...|    0|[3.83233030863620...|[0.97880008629612...|       0.0|
    |[31.0,5387.75,0.0...|    0|[3.24742811458119...|[0.96258058552664...|       0.0|
    |[31.0,7073.61,0.0...|    0|[3.79911450433881...|[0.97809976923405...|       0.0|
    |[31.0,11297.57,1....|    1|[0.79751152640735...|[0.68944192100551...|       0.0|
    |[31.0,11743.24,0....|    0|[7.95951793845681...|[0.99965080051155...|       0.0|
    |[31.0,12264.68,1....|    0|[3.77281170068563...|[0.97752920495855...|       0.0|
    |[32.0,6367.22,1.0...|    0|[3.20017220414578...|[0.96084075703562...|       0.0|
    |[32.0,8575.71,0.0...|    0|[4.52857300143358...|[0.98931923918898...|       0.0|
    |[32.0,13630.93,0....|    0|[2.65527248795398...|[0.93433521477806...|       0.0|
    |[33.0,4711.89,0.0...|    0|[7.15048703176813...|[0.99921613300884...|       0.0|
    |[33.0,5738.82,0.0...|    0|[5.41122451678732...|[0.99555369000330...|       0.0|
    |[33.0,7750.54,1.0...|    0|[4.79456321095382...|[0.99179329500352...|       0.0|
    |[33.0,12638.51,1....|    0|[4.15248449384766...|[0.98451815808214...|       0.0|
    |[33.0,13314.19,0....|    0|[3.36990907218523...|[0.96675076852634...|       0.0|
    |[34.0,5447.16,1.0...|    0|[3.75995719191832...|[0.97724510462861...|       0.0|
    |[34.0,6461.86,1.0...|    0|[4.80281076454080...|[0.99186015320798...|       0.0|
    |[34.0,7818.13,0.0...|    0|[4.73016790727597...|[0.99125221001613...|       0.0|
    |[34.0,9265.59,0.0...|    0|[4.83050636756087...|[0.99208073716831...|       0.0|
    +--------------------+-----+--------------------+--------------------+----------+
    only showing top 20 rows
    
    

### Using AUC


```python
churn_eval = BinaryClassificationEvaluator(rawPredictionCol='prediction',
                                           labelCol='churn')
```


```python
auc = churn_eval.evaluate(pred_and_labels.predictions)
```


```python
auc
```




    0.6866883116883117



[Common question - what is a good AUC value?](https://stats.stackexchange.com/questions/113326/what-is-a-good-auc-for-a-precision-recall-curve)

### Predict on brand new unlabeled data

We still need to evaluate the new_customers.csv file!


```python
final_lr_model = lr_churn.fit(final_data)
```


```python
new_customers = spark.read.csv('new_customers.csv',inferSchema=True,
                              header=True)
```


```python
new_customers.printSchema()
```

    root
     |-- Names: string (nullable = true)
     |-- Age: double (nullable = true)
     |-- Total_Purchase: double (nullable = true)
     |-- Account_Manager: integer (nullable = true)
     |-- Years: double (nullable = true)
     |-- Num_Sites: double (nullable = true)
     |-- Onboard_date: timestamp (nullable = true)
     |-- Location: string (nullable = true)
     |-- Company: string (nullable = true)
    
    


```python
test_new_customers = assembler.transform(new_customers)
```


```python
test_new_customers.printSchema()
```

    root
     |-- Names: string (nullable = true)
     |-- Age: double (nullable = true)
     |-- Total_Purchase: double (nullable = true)
     |-- Account_Manager: integer (nullable = true)
     |-- Years: double (nullable = true)
     |-- Num_Sites: double (nullable = true)
     |-- Onboard_date: timestamp (nullable = true)
     |-- Location: string (nullable = true)
     |-- Company: string (nullable = true)
     |-- features: vector (nullable = true)
    
    


```python
final_results = final_lr_model.transform(test_new_customers)
```


```python
final_results.select('Company','prediction').show()
```

    +----------------+----------+
    |         Company|prediction|
    +----------------+----------+
    |        King Ltd|       0.0|
    |   Cannon-Benson|       1.0|
    |Barron-Robertson|       1.0|
    |   Sexton-Golden|       1.0|
    |        Wood LLC|       0.0|
    |   Parks-Robbins|       1.0|
    +----------------+----------+
    
    


