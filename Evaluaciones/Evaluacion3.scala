// 1. Importamos una simple sesión spark
import org.apache.spark.sql.SparkSession

// 2. Implementamos líneas para minimizar errores
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

// 3. Creamos la instancia de sesion Spark
val spark = SparkSession.builder().getOrCreate()

// 4. Importamos librería KMeans para el algoritmo de agrupamiento
import org.apache.spark.ml.clustering.KMeans

// 5. Cargamos el archivo "Wholesale_customers_data.csv"
val data = spark.read.format("csv").option("inferSchema","true").option("header","true").csv("Wholesale_customers_data.csv")

//Imprimimos con printschema la estructura del dataset
data.printSchema()

// 6. Seleccionamos las columnas: Fresh, Milk, Grocery, Frozen, Detergents_Paper,Delicassen y llamamos al conjunto feature_data
val feature_data = data.select("Fresh","Milk","Grocery","Frozen","Detergents_Paper","Delicassen")

// 7. Importamos las librerías para transformación de los datos
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.feature.StringIndexer

// 8. Creamos un nuevo objeto Vector Assembler para las columnas de caracteristicas como un
// conjunto de entrada, recordando que no hay etiquetas.
// Este objeto almacenara las caracteristicas de la columna features   
val assembler = new VectorAssembler().setInputCols(Array("Fresh","Milk","Grocery","Frozen","Detergents_Paper","Delicassen")).setOutputCol("features")

// 9. Utilizamos el objeto assembler para transformar feature_data
val featureSet = assembler.transform(feature_data)

// 10. Creamos un modelo Kmeans con K=3
val kmeans = new KMeans().setK(3).setSeed(1L)
val model = kmeans.fit(featureSet)

// 11.Se Evalúan los grupos utilizando Within Set Sum of Squared Errors WSSSE
val WSSSE = model.computeCost(featureSet)
println(s"Within Set Sum of Squared Errors = $WSSSE")

// Se imprimen los centroides
println("Cluster Centers: ")
model.clusterCenters.foreach(println)

