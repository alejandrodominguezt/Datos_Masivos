# PRACTICES UNIT 2 (Description)
### Domínguez Tabardillo David Alejandro 15211698
### Saúl Soto Pino 15211705

**PRACTICES INDEX**

* [1.-Practice 1: Linear regression](#practice1)

**TASK INDEX**

* [1.-Task 1: Types of learning](#task1)
* [2.-Task 2: VectorAssembler and Vectors](#task2)


<a name="practice1"></a>

## Practice 1 

**LINEAR REGRESSION EXERCISE**

Instructions: Complete the commented tasks

```
// Import linear regression
    // Spark session and linear regression expression imported
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.regression.LinearRegression

// Optional: Use the following code to configure errors
    // Code was used to configure errors
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

// Start a simple Spark Session
val spark = SparkSession.builder().getOrCreate()

// Use Spark for the csv file Clean-Ecommerce.
    // Once the .csv file was loaded, it was used
val data  = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("Clean-Ecommerce.csv")

// Print the schema in the DataFrame.
    // The schema was printed to see the fields that the .csv file is made of
data.printSchema

// Print a sample line from the DataFrame.
    // head was used to print the head of the DataFrame
data.head(1)

//////////////////////////////////////////////////////
//// Configure el DataFrame para Machine Learning ////
//////////////////////////////////////////////////////

// Transforme el data frame para que tome la forma de
// ("label","features")
val colnames = data.columns
val firstrow = data.head(1)(0)
println("\n")
println("Example data row")
for(ind <- Range(1, colnames.length)){
    println(colnames(ind))
    println(firstrow(ind))
    println("\n")
}

// Import VectorAssembler and Vectors
    // Both were imported and the columns were shown at the time.
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

data.columns

// Rename the Yearly Amount Spent column as "label"
// Also from the data take only the numeric column
// Leave all this as a new DataFrame called df
    // The new DataFrame was created and only the non-String columns were taken
val df = data.select(data("Yearly Amount Spent").as("label"), 
$"Avg Session Length", $"Time on App", $"Time on Website", $"Length of Membership")

// Have the assembler object convert the input values to a vector, use the VectorAssembler 
// object to convert the input columns of the df
// to a single output column of an array named "features"
// Set the input columns from where we are supposed to read the values.
// Call this new assembler.




// Use the assembler to transform our DataFrame to two columns: label and features



// Create an object for linear regression model.



// Fit the model for the data and call this model lrModel




// Print the coefficients and intercept for the linear regression



// Summarize the model on the training set print the output of some metrics!
// use our model's .summary method to create an object called trainingSummary
// Show the residuals values, the RMSE, the MSE, and also the R ^ 2.




```


<a name="task1"></a>


## Task 1 

**Types of learning**

Instructions: Investigate the 3 types of learning scala

### 1. Supervised machine learning
In the supervised learning algorithms, a predictive model is generated, based on input and output data. 
The keyword “supervised” comes from the idea of having a previously labeled and classified data set, 
that is, having a sample set which already knows which group, value or category the examples belong to. 
With this group of data, which we call training data, the adjustment is made to the initial model proposed. 
In this way, the algorithm "learns" how to classify the input samples by comparing the result of the model, 
and the actual label of the sample, making the respective compensations for the model according to each error 
in the estimation of the result. For example, supervised learning has been used for programming autonomous vehicles. 
Some methods and algorithms that we can implement are the following:
  * K-nearest neighbors
  * Artificial neural networks
  * Support vector machines
  * Naive Bayes classifier
  * Decision trees
  * Logistic regression

### 2. Unsupervised machine learning
Unsupervised learning algorithms work very similarly to supervised ones, with the difference that they only adjust 
their predictive model taking into account the input data, regardless of the output data. That is, unlike the supervised, 
the input data is neither classified nor labeled, and these characteristics are not necessary to train the model. Within 
this type of algorithms, grouping or clustering in English is the most widely used, since it divides the data into groups 
that have similar characteristics to each other. One application of these methods is image compression. Among the main 
algorithms of unsupervised type stand out:

  * K-means
  * Gaussian mixtures
  * Hierarchical clustering
  * Self-organizing maps

### 3. Reinforcement learning
Reinforcement learning algorithms define models and functions focused on maximizing a measure of "rewards", based on "actions" and the environment in which the intelligent agent will perform.

This algorithm is the most attached to human behavioral psychology, since it is an action-reward model, which seeks to make the algorithm fit the best "reward" given by the environment, and its actions to be taken are subject to these rewards.

These kinds of methods can be used to make robots learn to perform different tasks. Among the most used algorithms we can name:
*Dynamic programming
*Q-learning
*SARSA

For more information: [Machine learning types](https://medium.com/soldai/tipos-de-aprendizaje-autom%C3%A1tico-6413e3c615e2)

<a name="task2"></a>


## Task 2

**VectorAssembler and Vectors**

Instructions: investigate what it does 

### 1. Reinforcement learning
Investigate what VectorAssembler and Vectors do
**VectorAssembler** 
It is a transformer that combines a given list of columns into a single vector column. 
It is useful for combining raw features and features generated by different feature transformers into a 
single feature vector, in order to train ML models such as logistic regression and decision trees. 
VectorAssembler accepts the following types of input columns: all numeric types, boolean type and vector type. 
In each row, the values in the input columns are concatenated into a vector in the specified order.

**Vectors:** 
Represents a numeric vector, whose index type is Int and the value type is Double.



### 2. Search documentation how to calculate rootMeanSquaredError
RMSE: It is the square root of the variance.

## Task 3

**Pipeline and Confusion Matrix**

**Pipeline**
Introduced in Spark 1.2, the Pipeline API is a high-level API for MLlib. The concept of Pipelines is to facilitate the creation, adjustment and inspection of practical ML workflows. In other words, it allows us to focus more on solving a machine learning task, rather than wasting the time spent organizing the code.

A Spark Pipeline is specified as a sequence of stages, and each stage is either a transformer or an estimator. These stages are executed in order, and the input Data Frame is transformed as it goes through each stage.

**Transformers**
A transformer is an abstraction that includes transformers of features and learned models. Technically, a transformer implements a transform () method, which converts one DataFrame to another, generally adding one or more columns. For example:

A feature transformer can take a DataFrame, read a column (eg, Text), assign it to a new column (eg, Feature Vectors), and generate a new Data Frame with the assigned column added.

A learning model can take a DataFrame, read the column containing the feature vectors, predict the label for each feature vector, and generate a new Data Frame with the predicted labels added as a column.

**Estimators**
An estimator abstracts the concept of a learning algorithm or any algorithm that fits or trains on data. Technically, an Estimator implements a fit () method, which accepts a DataFrame and produces a Model, which is a Transformer. For example, a learning algorithm like Logistic Regression is an Estimator, and calling fit () trains a Logistic Regression Model, which is a Model and therefore a Transformer.

For more information: [Insight Data Science: Blog de Pipelines en Spark ](https://blog.insightdatascience.com/spark-pipelines-elegant-yet-powerful-7be93afcdd42)

**Confusion Matrix**
A confusion matrix is a performance measure for the machine learning classification problem where the output can be two or more classes. It is a table with 4 different combinations of predicted and actual values.

![alt text](https://miro.medium.com/max/712/1*Z54JgbS4DUwWSknhDCvNTQ.png "Confusion Matrix")

It is extremely useful for measuring recovery, precision, specificity, precision, and most importantly, the AUC-ROC curve.

Let's understand TP, FP, FN, TN in terms of pregnancy analogy.

![alt text](https://miro.medium.com/max/924/1*7EYylA6XlXSGBCF77j_rOA.png "Confusion Matrix Example")

True Positive:
Interpretation: You predicted positive and it's true.
You predicted that a woman is pregnant and actually is.

True negative:
Interpretation: You predicted negative and it's true.
You predicted that a man is not pregnant and actually is not.

False positive: (type 1 error)
Interpretation: You predicted positive and it is false.
You predicted that a man is pregnant but in fact is not.

False negative: (type 2 error)
Interpretation: You predicted negative and it is false.
You predicted that a woman is not pregnant, but in fact she is.

Just remember, we describe the predicted values as positive and negative and the actual values as true and false.

![alt text](https://miro.medium.com/max/880/1*2lptVD05HarbzGKiZ44l5A.png "Results")

For more information: [Towards Data Science: Understanding Confusion Matrix  ](https://towardsdatascience.com/understanding-confusion-matrix-a9ad42dcfd62)