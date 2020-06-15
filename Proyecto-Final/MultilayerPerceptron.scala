// Import of used libraries
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.DateType
import org.apache.spark.sql.{SparkSession, SQLContext}
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.log4j._
// Reduce errors
Logger.getLogger("org").setLevel(Level.ERROR)
// A spark session was created and the .csv file was loaded
val spark = SparkSession.builder().getOrCreate()
val df = spark.read.option("header","true").option("inferSchema","true").option("delimiter",";").format("csv").load("bank-full.csv")
// We print the schema of our dataset
df.printSchema()
// We show the records of the first row
df.show(1)
// We change the columns to binary data.
val col1 = df.withColumn("y",when(col("y").equalTo("yes"),1).otherwise(col("y")))
val col2 = col1.withColumn("y",when(col("y").equalTo("no"),2).otherwise(col("y")))
val newcol = col2.withColumn("y",'y.cast("Int"))
// Mostramos la primera fila con datosbinarios
newcol.show(1)
// We create the characteristics table
val assembler = new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features")
val featur = assembler.transform(newcol)
// We show the new resulting column
featur.show(1)

// We make the change of the column "y" by label
val colchange = featur.withColumnRenamed("y", "label")
val feat = colchange.select("label","features")
// We show the new label column
feat.show(1)
// We perform a necessary split for Multilayer perceptron
// We divide the data in an array into parts of 60% and 40%
val split = feat.randomSplit(Array(0.6, 0.4), seed = 1234L)
val train = split(0)
val test = split(1)
// We create an array to perform input and output tests of the features
val layers = Array[Int](5, 2, 2, 4)
// We implemented the Perceptron Multilayer classification with a maximum of 100 iterations
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(10)
// We train and adjust the model
val model = trainer.fit(train)
// We print the model accuracy
val result = model.transform(test)
val predictionAndLabels = result.select("prediction", "label")
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")

// We show the time the process took
val time = System.nanoTime
val duration = (System.nanoTime - time) / 1e9d
