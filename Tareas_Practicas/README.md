# PRACTICES UNIT 2 (Description)
### Domínguez Tabardillo David Alejandro 15211698
### Saúl Soto Pino 15211705

**PRACTICES INDEX**

* [Practice 1: Linear regression](#practice1)
* [Practice 2: Logistic regression](#practice2)
* [Practice 3: Desicion Trees Clasification](#practice2)
* [Practice 4: Random Forest](#practice2)
* [Practice 5: Gradient Boosted Trees (GBTs)](#practice2)
* [Practice 6: Multilayer perceptron classifier](#practice2)
* [Practice 7: Linear Support Vector Machine (SVM)](#practice2)
* [Practice 8: One-vs-Rest classifier (a.k.a. One-vs-All)](#practice2)

**TASK INDEX**

* [Task 1: Types of learning](#task1)
* [Task 2: VectorAssembler and Vectors](#task2)
* [Task 3: Pipeline and Confusion Matrix](#task2)


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
    // A new Assembler named nuevoassembler was created taking into account the data to read
    // and this was placed in the feautres column
val nuevoassembler = new VectorAssembler().setInputCols(Array("Avg Session Length", "Time on App", "Time on Website", "Length of Membership")).setOutputCol("features")

// Use the assembler to transform our DataFrame to two columns: label and features
   // The new assembler was transformed to two columns and they were shown with show
val output = nuevoassembler.transform(df).select($"label", $"features")
output.show()

// Create an object for linear regression model.
val lr = new LinearRegression()

// Fit the model for the data and call this model lrModel
val lrModelo = lr.fit(output)

// Print the coefficients and intercept for the linear regression
val trainingSummary = lrModelo.summary
trainingSummary.residuals.show()

// Summarize the model on the training set print the output of some metrics!
// use our model's .summary method to create an object called trainingSummary
// Show the residuals values, the RMSE, the MSE, and also the R ^ 2.
trainingSummary.predictions.show()
trainingSummary.r2 //variaza que hay 
trainingSummary.rootMeanSquaredError

```
<a name="practice2"></a>


## Practice 2

**Logistic regression**

In this project, we will work with a set of false advertising data, indicating whether a particular internet user clicked on an ad.
We will try to create a model that predicts whether or not they will click on an ad based on the characteristics of that user.
This dataset contains the following characteristics:

- 'Daily time spent on site': consumer time on site in minutes

- 'Age': age of the client in years

- 'Area income': Avg. Consumer's geographical area income

- 'Daily use of the Internet': Average minutes per day the consumer is on the Internet

- 'Ad topic line': ad title

- 'City': consumer city

- 'Man': whether or not the consumer was a man

- 'Country': consumer country

- 'Timestamp': time the consumer clicked on the ad or in the closed window

- 'You clicked on the ad': 0 or 1 indicated clicking on the ad

Complete las siguientes tareas que están comentas 
```
////////////////////////
/// Take the data //////
//////////////////////

// Import a SparkSession with the Logistic Regression library
         // Imported the Logistic Regression library and the Spark session
         
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SparkSession

// Optional: Use the Error reporting code
         // The Log4j library was used to write log messages
         // in this case to report an error

import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

// Create a Spark session
     // A simple spark session was created

val spark = SparkSession.builder().getOrCreate()

// Use Spark to read the csv Advertising file.
         // The advertisign.csv file was imported to the value data

val data  = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("advertising.csv")

// Print the Schema of the DataFrame
     // The schema of the value data was printed, which was imported with the advertising file

data.printSchema()

///////////////////////
/// Deploy the data /////
/////////////////////

// Print an example row
     // Print the head (first record of the imported file)

data.head(1)

val colnames = data.columns
val firstrow = data.head(1)(0)
println("\n")
println("Example data row")
for(ind <- Range(1, colnames.length)){
    println(colnames(ind))
    println(firstrow(ind))
    println("\n")
}

////////////////////////////////////////////////////
//// Prepare the Data Frame for Machine Learning ////
//////////////////////////////////////////////////

//   Do the next:
// - Rename the column "Clicked on Ad" to "label"
// - Take the following columns as features "Daily Time Spent on Site", "Age", "Area Income", "Daily Internet Usage", "Timestamp", "Male"
// - Create a new column called "Hour" of the Timestamp containing the "Hour of the click"
         // The "Hour" column of the Timestamp was created

val timedata = data.withColumn("Hour",hour(data("Timestamp")))

// The "Clicked on Ad" column was renamed to "label" using a select from the timedata
// The following columns were taken as features "Daily Time Spent on Site", "Age", "Area Income", "Daily Internet Usage", "Timestamp", "Male"

val logregdata = timedata.select(data("Clicked on Ad").as("label"), $"Daily Time Spent on Site", $"Age", $"Area Income", $"Daily Internet Usage", $"Hour", $"Male")

// Import VectorAssembler and Vectors
         // Vectors and VectorAssembler were imported to allow merging vectors

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

// Create a new VectorAssembler object called assembler for the features
// The VectorAssembler object called assembler was created for the columns of the array and its columns
val assembler = (new VectorAssembler()
                  .setInputCols(Array("Daily Time Spent on Site", "Age","Area Income","Daily Internet Usage","Hour","Male"))
                  .setOutputCol("features"))

// Use randomSplit to create train and test data divided by 70/30
         // The data was divided to 70% for training and 30% for tests

val Array(training, test) = logregdata.randomSplit(Array(0.7, 0.3), seed = 12345)

///////////////////////////////
//  Set up a Pipeline  ///////
/////////////////////////////

// Amount Pipeline
// Create a new LogisticRegression object called lr
// Create a new pipeline with the elements: assembler, lr
// Adjust the pipeline for the training set.
// Take the Results in the Test set with transform

         // Pipeline imported
 
import org.apache.spark.ml.Pipeline
        
    // A new element for Logistic Regression was created called lr

val lr = new LogisticRegression()

    // A new pipeline called pipeline is created with the assembler and lr elements

val pipeline = new Pipeline().setStages(Array(assembler, lr))
       
    // The new pipe for the training set was fitted to a model
val model = pipeline.fit(training)
        
        // The result was assigned to the transformed model of the Test set

val results = model.transform(test)

////////////////////////////////////
//// Model evaluation /////////////
//////////////////////////////////

// For Metrics and Evaluation import MulticlassMetrics
// Convert test results into RDD using .as and .rdd
// Initialize a MulticlassMetrics object
// Print the Confusion matrix

        // MulticlassMetrics libraries were imported

import org.apache.spark.mllib.evaluation.MulticlassMetrics
       
        // A selection was made to the results of the Test model with the columns "predict
         // To make them Double
         // RDD is a fault tolerant collection of elements and operates in parallel

val predictionAndLabels = results.select($"prediction",$"label").as[(Double, Double)].rdd

       // An object called MulticlassMetrics of the converted data was provided
         // which is assigned to the value metrics

val metrics = new MulticlassMetrics(predictionAndLabels)
     
  // The confusion matrix was printed
println("Confusion matrix:")
println(metrics.confusionMatrix)
metrics.accuracy


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
 * Dynamic programming
 * Q-learning
 * SARSA

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

<a name="task3"></a>

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


<a name="practice3"></a>
## Practice 3

**Decision tree classifier**

Decision trees are a popular family of classification and regression methods. More information about the spark.ml implementation can be found further in the section on decision trees.

* Examples

*The following examples load a dataset in LibSVM format, split it into training and test sets, train on the first dataset, and then evaluate on the held-out test set. We use two feature transformers to prepare the data; these help index categories for the label and categorical features, adding metadata to the DataFrame which the Decision Tree algorithm can recognize.*

```

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

// Load the data stored in LIBSVM format as a DataFrame.
val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

// Index labels, adding metadata to the label column.
// Fit on whole dataset to include all labels in index.
val labelIndexer = new StringIndexer()
  .setInputCol("label")
  .setOutputCol("indexedLabel")
  .fit(data)
// Automatically identify categorical features, and index them.
val featureIndexer = new VectorIndexer()
  .setInputCol("features")
  .setOutputCol("indexedFeatures")
  .setMaxCategories(4) // features with > 4 distinct values are treated as continuous.
  .fit(data)

// Split the data into training and test sets (30% held out for testing).
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

// Train a DecisionTree model.
val dt = new DecisionTreeClassifier()
  .setLabelCol("indexedLabel")
  .setFeaturesCol("indexedFeatures")

// Convert indexed labels back to original labels.
val labelConverter = new IndexToString()
  .setInputCol("prediction")
  .setOutputCol("predictedLabel")
  .setLabels(labelIndexer.labels)

// Chain indexers and tree in a Pipeline.
val pipeline = new Pipeline()
  .setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))

// Train model. This also runs the indexers.
val model = pipeline.fit(trainingData)

// Make predictions.
val predictions = model.transform(testData)

// Select example rows to display.
predictions.select("predictedLabel", "label", "features").show(5)

// Select (prediction, true label) and compute test error.
val evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("indexedLabel")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println(s"Test Error = ${(1.0 - accuracy)}")

val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
println(s"Learned classification tree model:\n ${treeModel.toDebugString}")


Source: http://spark.apache.org/docs/latest/ml-classification-regression.html#decision-tree-classifier

```

<a name="practice4"></a>
## Practice 4

**Random Forest**

```



```





<a name="practice5"></a>
## Practice 5

**Gradient Boosted Trees (GBTs)**


```

// We import our libraries to use ml
// pipeline to process the data and join it
// GBT for classification and modeling
// Feature to index our data

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

// Spark Session 
import org.apache.spark.sql.SparkSession

import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

// We load our dataset from mlib
val data = spark.read.format("libsvm").load("sample_libsvm_data.txt")
data.printSchema()

// We index our labels to work them
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)

// Automatically identify categories and index them
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)

// We divide our data set into training data and test or testing data
// 70% for training and% 30 for testing.
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

// We train our GBT model.
val gbt = new GBTClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setMaxIter(10).setFeatureSubsetStrategy("auto")

// We turn our indexed tags into original tags
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

// we join our gbt model our indexed labels through Pipeline
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, gbt, labelConverter))

// We train our pipeline using training data and the fit function
val model = pipeline.fit(trainingData)

// We create our predictions transforming our data
val predictions = model.transform(testData)

// select some test columns and display them.
predictions.select("predictedLabel", "label", "features").show(5)

// We select our predictions and true labels as well as the margin of error obtained.
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")

// Evaluations and we obtain the percentage of the prediction
val accuracy = evaluator.evaluate(predictions)
println(s"Test Error = ${1.0 - accuracy}")

// Print result of Trees using GBTs
val gbtModel = model.stages(2).asInstanceOf[GBTClassificationModel]
println(s"Learned classification GBT model:\n ${gbtModel.toDebugString}")

```


<a name="practice6"></a>
## Practice 6

**Multilayer perceptron classifier**

```



```

<a name="practice7"></a>
## Practice 7

**Linear Support Vector Machine (SVM)**

```
// We import libraries

package org.apache.spark.examples.ml

//Library for SVC

import org.apache.spark.ml.classification.LinearSVC

import org.apache.spark.sql.SparkSession

// Create session spark variable

val spark = SparkSession.builder.appName("LinearSVCExample").getOrCreate()

// Load the data from our file and add it to a variable to train

val training = spark.read.format("libsvm").load("/usr/local/spark-2.3.4-bin-hadoop2.6/data/mllib/sample_libsvm_data.txt")

// create an object of type LinearSVC, Set the number of iterations to 10 with the setMaxIter method and Set the regularization parameter

val lsvc = new LinearSVC().setMaxIter(10).setRegParam(0.1)

// We fit the model

val lsvcModel = lsvc.fit(training)

// We print the coefficients of the vector and its intercept

println(s"Coefficients: ${lsvcModel.coefficients} Intercept: ${lsvcModel.intercept}")



```


<a name="practice8"></a>
## Practice 8

**One-vs-Rest classifier (a.k.a. One-vs-All)**

```
// Import Libraries for OneVSRest
import org.apache.spark.ml.classification.{LogisticRegression, OneVsRest}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

// load data file.
val inputData = spark.read.format("libsvm")
  .load("data/mllib/sample_multiclass_classification_data.txt")

// generate the train/test split.
val Array(train, test) = inputData.randomSplit(Array(0.8, 0.2))

// instantiate the base classifier
val classifier = new LogisticRegression()
  .setMaxIter(10)
  .setTol(1E-6)
  .setFitIntercept(true)

// instantiate the One Vs Rest Classifier.
val ovr = new OneVsRest().setClassifier(classifier)

// train the multiclass model.
val ovrModel = ovr.fit(train)

// score the model on test data.
val predictions = ovrModel.transform(test)

// obtain evaluator.
val evaluator = new MulticlassClassificationEvaluator()
  .setMetricName("accuracy")

// compute the classification error on test data.
val accuracy = evaluator.evaluate(predictions)
println(s"Test Error = ${1 - accuracy}")

```
