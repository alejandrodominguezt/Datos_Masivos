// Import of used libraries
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
// Reduce errors
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

// A spark session was created and the .csv file was loaded
val spark = SparkSession.builder().getOrCreate()
val data = spark.read.option("header","true").option("inferSchema","true").option("delimiter",";").format("csv").load("bank-full.csv")
// We print the schema of our dataset
df.printSchema()
// We show the records of the first row
df.show(1)

// We change the columns to binary data.
val cbin1 = data.withColumn("y",when(col("y").equalTo("yes"),1).otherwise(col("y")))
val cbin2 = cbin1.withColumn("y",when(col("y").equalTo("no"),2).otherwise(col("y")))
val cnew = cbin2.withColumn("y",'y.cast("Int"))

// We show the first row with binary data
cnew.show(1)

// We create the characteristics table
val assembler = new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features")
val featur= assembler.transform(cnew)

// We show the new resulting column
featur.show(1)

// We make the change of the column "y" by label
val cambio = featur.withColumnRenamed("y", "label")
val feat = cambio.select("label","features")
// Mostramos la nueva columna label
feat.show(1)

// We carry out the logistic regression process
// With 10 iterations
// We divide our set into 30% and 80%
val logistic = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)

// Realizamos el ajuste del modelo
val logisticModel = logistic.fit(feat)
// Imprimimos los coeficientes y loas intersecciones del modelo
println(s"Coefficients: ${logisticModel.coefficients} Intercept: ${logisticModel.intercept}")
val logisticMult = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8).setFamily("multinomial")
val logisticMultModel = logisticMult.fit(feat)
println(s"Multinomial coefficients: ${logisticMultModel.coefficientMatrix}")
println(s"Multinomial intercepts: ${logisticMultModel.interceptVector}")

// We show the time the process took
val time = System.nanoTime
val duration = (System.nanoTime - time) / 1e9d
 
