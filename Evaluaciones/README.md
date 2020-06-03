**Evaluation 2:**

### Evaluation 2

// 1. a) Mllib library was imported from Spark the Machine Learning algorithm called multilayer perceptron
```
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
```

// Other necessary libraries were imported
```
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
```

// Import VectorAssembler and Vectors
```
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.ml.linalg.Vectors
```

// Start a simple spark session
```
import org.apache.spark.sql.SparkSession
```

// The data from the dataset "iris.csv" is loaded
```
val data = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("iris.csv")
```

//2. The function "columns" is used to know the names of the columns
```
data.columns
```

//3. Outline of data is printed
```
data.printSchema()
```

//4. Show was used to print the first 5 columns of the dataset
```
data.show(5)
```

//5. Describe () was used to learn more about the dataframe and show () was used to display this data.
```
data.describe().show()
```

// Dataclean was used to remove null fields
```
val dataClean = data.na.drop()
```

// Transform the data to the variable "features" using the VectorAssembler library
```
val vectorFeatures = (new VectorAssembler().setInputCols(Array("sepal_length","sepal_width", "petal_length","petal_width")).setOutputCol("features"))
```

// The features are transformed using the dataframe
```
val features = vectorFeatures.transform(dataClean)
```

// The new object "StringIndexer" was declared which will transform the data to numeric data
```
val speciesIndexer = new StringIndexer().setInputCol("species").setOutputCol("label")
```
// We adjust the indexed species with the vector features
```
val dataIndexed = speciesIndexer.fit(features).transform(features)
```

// With the variable "splits" we make a cut randomly
// The dataset is divided into 60% for training and 40% for testing
```
val splits = dataIndexed.randomSplit(Array(0.6, 0.4), seed = 1234L)
```

// Declare the variable "train" which will have 60% of the data
```
val train = splits(0)
```
// The variable "test" is declared which will have 40% of the data
```
val test = splits(1)
```
// Random training tests are done to train the algorithm and build the model
```
val layers = Array[Int](4, 5, 4, 3)
```

// Trainer is created with the MultiplayerPerceptronClassifier library, configure the Multilayer algorithm trainer with its parameters
```
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)
```
// Train the model by adjusting the training data
```
val model = trainer.fit(train)
```

// The model already trained with the tests is transformed
```
val result = model.transform(test)
```

// Select the prediction and the label that will be stored in the variable
```
val predictionAndLabels = result.select("prediction", "label")
```

// Some data is displayed
```
predictionAndLabels.show()
```

// The MulticlassClassificationEvaluator library is used with the labels assigned to the value evaluator
```
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")
```

// We get an estimate of the precision of the model
```
val accuracy = evaluator.evaluate(predictionAndLabels)
```
// Se imprime el error del modelo
```
println(s"Test Error = ${(1.0 - accuracy)}")
```

### Video: 
https://youtu.be/czo9Wh7dwW8
