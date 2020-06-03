import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

val data = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("iris.csv")

data.columns

data.printSchema()

data.show(5)

data.describe().show()

val dataClean = data.na.drop()

val vectorFeatures = (new VectorAssembler().setInputCols(Array("sepal_length","sepal_width", "petal_length","petal_width")).setOutputCol("features"))

val features = vectorFeatures.transform(dataClean)

val speciesIndexer = new StringIndexer().setInputCol("species").setOutputCol("label")

val dataIndexed = speciesIndexer.fit(features).transform(features)

val splits = dataIndexed.randomSplit(Array(0.6, 0.4), seed = 1234L)

val train = splits(0)

val test = splits(1)

val layers = Array[Int](4, 5, 4, 3)

val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)

val model = trainer.fit(train)

val result = model.transform(test)

val predictionAndLabels = result.select("prediction", "label")

predictionAndLabels.show()

val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")

val accuracy = evaluator.evaluate(predictionAndLabels)

println(s"Test Error = ${(1.0 - accuracy)}")
