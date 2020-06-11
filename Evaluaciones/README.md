TEAM
* Domínguez Tabardillo David Alejandro 15211698
* Saúl Soto Pino 15211705

## Evaluation Unit 3
```
// 1. We import a simple spark session
import org.apache.spark.sql.SparkSession

// 2. We implement lines to minimize errors
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

// 3. We create the instance of session Spark
val spark = SparkSession.builder().getOrCreate()

// 4. We import KMeans library for the grouping algorithm
import org.apache.spark.ml.clustering.KMeans

// 5. We load the file "Wholesale_customers_data.csv"
val data = spark.read.format("csv").option("inferSchema","true").option("header","true").csv("Wholesale_customers_data.csv")

// Print with printschema the structure of the dataset
data.printSchema()

// 6. We select the columns: Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen and call the set feature_data
val feature_data = data.select("Fresh","Milk","Grocery","Frozen","Detergents_Paper","Delicassen")

// 7. We import the libraries for data transformation
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.feature.StringIndexer

// 8. We create a new Vector Assembler object for feature columns as a
// input set, remembering that there are no labels.
// This object will store the characteristics of the features column
val assembler = new VectorAssembler().setInputCols(Array("Fresh","Milk","Grocery","Frozen","Detergents_Paper","Delicassen")).setOutputCol("features")

// 9. We use the assembler object to transform feature_data
val featureSet = assembler.transform(feature_data)

// 10. We create a Kmeans model with K = 3
val kmeans = new KMeans().setK(3).setSeed(1L)
val model = kmeans.fit(featureSet)

// 11.The groups are evaluated using Within Set Sum of Squared Errors WSSSE
val WSSSE = model.computeCost(featureSet)
println(s"Within Set Sum of Squared Errors = $WSSSE")

// The centroids are printed
println("Cluster Centers: ")
model.clusterCenters.foreach(println)

```
Video: https://youtu.be/xRGqjE9Xx9E
