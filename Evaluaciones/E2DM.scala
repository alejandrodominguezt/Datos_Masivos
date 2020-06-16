// 1. a) Se importó la librería Mllib de Spark el algoritmo de Machine Learning llamado multilayer perceptron
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
// Se importaron otras librerías necesarias
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
// Se Importa VectorAssembler y Vectors
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.ml.linalg.Vectors
// iniciar una simple sesión spark
import org.apache.spark.sql.SparkSession
// Se cargan los datos del dataset "iris.csv"
val data = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("iris.csv")
//2. Se usa la función "columns"" para saber los nombres de las columnas
data.columns
//3. Se imprime el esquema de los datos 
data.printSchema()
//4. Se utilizó head para imprimir las primeras 5 columnas del dataset
data.show(5)
//5. Se usó describe() para aprender más sobre el dataframe y show() para mostrar estos datos
data.describe().show()
// Se uso dataclean para eliminar los campos nulos
val dataClean = data.na.drop()
// Se transforman los datos a la variable "features" usando la libreria VectorAssembler
val vectorFeatures = (new VectorAssembler().setInputCols(Array("sepal_length","sepal_width", "petal_length","petal_width")).setOutputCol("features"))
// Se transforman los features usando el dataframe
val features = vectorFeatures.transform(dataClean)
// Se declaró el nuevo objeto "StringIndexer" que transformará los datos a datos numericos
val speciesIndexer = new StringIndexer().setInputCol("species").setOutputCol("label")
// Ajustamos las especies indexadas con el vector features
val dataIndexed = speciesIndexer.fit(features).transform(features)
// Con la variable "splits" hacemos un corte de forma aleatoria
// El conjunto de datos es dividido en 60% para entrenamiento y 40% para prueba
val splits = dataIndexed.randomSplit(Array(0.6, 0.4), seed = 1234L)
// Se declara la variable "train" la cual tendra el 60% de los datos
val train = splits(0)
// Se declara la variable "test" la cual tendra el 40% de los datos
val test = splits(1)
// Se hacen pruebas de entrenamiendo Random para entrenar el algoritmo y construir el modelo
val layers = Array[Int](4, 5, 4, 3)
// Se crea trainer con la libreria MultiplayerPerceptronClassifier configura el entrenador del algoritmo Multilayer con sus parametros
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)
// Se entrena el modelo ajustando los datos de entrenamiento 
val model = trainer.fit(train)
// Se transforma el modelo ya entrenado con las purebas
val result = model.transform(test)
// Se selecciona la prediccion y la etiqueta que seran guardado en la variable
val predictionAndLabels = result.select("prediction", "label")
// Se muestran algunos datos
predictionAndLabels.show()
// Se usa la libreria MulticlassClassificationEvaluator con las etiquetas asignadas al value evaluator
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")
// Obtenemos una estimación de la precisión del modelo
val accuracy = evaluator.evaluate(predictionAndLabels)
// Se imprime el error del modelo
println(s"Test Error = ${(1.0 - accuracy)}")
