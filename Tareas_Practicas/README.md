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
  *
  * 

### 3. Reinforcement learning







Reference: https://medium.com/soldai/tipos-de-aprendizaje-autom%C3%A1tico-6413e3c615e2

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


