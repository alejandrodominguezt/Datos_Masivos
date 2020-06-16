// LINEAR REGRESSION EXERCISE 
// Complete the commented tasks

// Import LinearRegression
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.regression.LinearRegression

// Optional: Use the following code to configure errors
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

// Start a simple Spark Session
val spark = SparkSession.builder().getOrCreate()

// Use Spark for the csv file Clean-Ecommerce.
// Once the .csv file was loaded, it was used
val data  = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("Clean-Ecommerce.csv")

// Print the schema in the DataFrame.
data.printSchema

// Print a sample line from the DataFrame.
// head was used to print the head of the DataFrame
data.head(1)

//////////////////////////////////////////////////////
//// Configure the DataFrame for Machine Learning ////
//////////////////////////////////////////////////////

// Transform the data frame so that it takes the form of
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
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

data.columns

// Rename the Yearly Amount Spent column as "label"
// Also from the data take only the numeric column
// Leave all this as a new DataFrame called df
val df = data.select(data("Yearly Amount Spent").as("label"), $"Avg Session Length", $"Time on App", $"Time on Website", $"Length of Membership")

// Have the assembler object convert the input values to a vector
// Use the VectorAssembler object to convert the input columns of the df
// to a single output column of an array named "features"
// Set the input columns from where we are supposed to read the values.
// Call this new assembler.

val nuevoassembler = new VectorAssembler().setInputCols(Array("Avg Session Length", "Time on App", "Time on Website", "Length of Membership")).setOutputCol("features")

// Use the assembler to transform our DataFrame to two columns: label and features

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
// Use our model's .summary method to create an object
// called trainingSummary

// Show the residuals values, the RMSE, the MSE, and also the R ^ 2.
trainingSummary.predictions.show()
trainingSummary.r2 //variaza que hay 
trainingSummary.rootMeanSquaredError
