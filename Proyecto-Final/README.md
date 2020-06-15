# Final Project (IEEE)
### Domínguez Tabardillo David Alejandro 15211698
### Saúl Soto Pino 15211705

![Grafica](https://github.com/alejandrodominguezt/Datos_Masivos/blob/development/cover.PNG?raw=true)

**INDEX**

* [Abstract](#abstract)
* [Keywords:](#keywords)
* [CHAPTER I.  Introduction](#chapter1)
* [CHAPTER II. objectives](#chapter2)
* [CHAPTER III. Theoretical Framework](#chapter3)
* [CHAPTER IV. Implementation](#chapter4)
* [CHAPTER V. Results](#chapter5)
* [CHAPTER VI. Conclusions](#chapter6)
* [References](#references)


<a name="abstract"></a>

## Abstract

This document presents a comparison of machine learning algorithms through implementation 
in the Scala language with Spark, as well as performance tests with iterations specified 
in a comparative table of results, in order to gain an in-depth understanding of the best 
and worst aspects of each one, in a demonstrative way for its application in daily aspects 
that require the use of large volumes of data. The algorithms studied are the following:

* SVM
* Decision tree
* Logistic regression
* Multilayer Perceptron

<a name="keywords"></a>

## Keywords
Machine Learning, Decision Trees, Logistic Regression, Support Vector Machines, Multilayer Perceptron.

<a name="chapter1"></a>
## CHAPTER I.  Introduction
Next, a comparison of 4 Machine Learning models studied in the Massive Data course will be 
carried out in order to publicize the performance of each in terms of precision and processing time.

Chapter ll contains the objectives to be met in carrying out this document, chapter lll includes 
all the information corresponding to each algorithm, considering its advantages and disadvantages, 
chapter lV shows the tools that will be used to process the data of each algorithm and to obtain 
the comparison, in chapter V the comparative tables and the averages obtained in the implementation 
are shown and finally in chapter V the conclusions of the work carried out are presented.

<a name="chapter2"></a>
## CHAPTER II.  Objectives

**General**
Next, a comparison of 4 Machine Learning models studied in the Massive Data course will be carried 
out in order to publicize the performance of each in terms of precision and processing time.

Study and learn more about machine learning algorithms by evaluating their performance using a bank 
data file, obtaining useful aspects for the application in the future.

**Specific**
* Implement a dataset for 4 different Machine Learning algorithms
* Perform 10 iterations of each algorithm studied to evaluate its performance.
* Measure the precision of each classifier to know its accuracy in each iteration.
* Demonstrate the effectiveness of each classifier according to error tests over time.
* Generate an average to reach a conclusion based on all the classifiers studied.
* Represent the results obtained in tabular form.

<a name="chapter3"></a>
## CHAPTER III.  Theoretical Framework

**SVM**
A support vector machine (SVM) is a discriminative classifier formally defined 
by a separation hyperplane.

In other words, given the labeled training data (supervised learning), the algorithm 
generates an optimal hyperplane that categorizes new examples. In two dimensional spaces, 
this hyperplane is a line that divides a plane into two parts where in each class it is 
located on each side. [3]

**Advantages**
* Effective in large spaces.
* It remains effective in cases where the number of dimensions is greater than the number of samples.
* It uses a subset of training points in the decision function (called support vectors), 
  so it is also memory efficient.
* Versatile: Different Kernel functions can be specified for the decision function. Common cores 
  are provided, but it is also possible to specify custom cores. [4]
  
  **Disadvantages**  
* If the number of features is much greater than the number of samples, avoid over-tuning when 
  choosing the core functions and the regularization term is crucial.
* SVMs do not directly provide probability estimates, they are calculated using an expensive 
  five-fold cross-validation. [4]

**Decision tree**
A decision tree is a decision support tool that uses a tree-like decision graph or model and its 
possible consequences, including fortuitous event results, resource costs, and profit. It is a way 
of displaying an algorithm that only contains conditional control statements. [5]

**Advantages**
* Easy to understand: The output of the decision tree is very easy to understand even for people 
  with a non-analytical background. It does not require any statistical knowledge to read and interpret 
  them. Its graphical representation is very intuitive and users can easily relate their hypothesis.
* Useful in data exploration: the decision tree is one of the fastest ways to identify the most s
  ignificant variables and the relationship between two or more variables. With the help of decision trees, 
  we can create new variables / characteristics that have better power to predict the target variable. It can also be used in the data     exploration stage. For example, we are working on a problem in which we have information available on hundreds of variables, there the   decision tree will help to identify the most significant variable.
* Decision trees implicitly carry out variable selection or feature selection.
* Decision trees require relatively little effort from users to prepare data.
* Less data cleaning required: Requires less data cleaning compared to some other modeling techniques. It is not influenced by outliers   and missing values to a fair degree.
* The data type is not a constraint: it can handle numerical and categorical variables. It can also handle multiple output problems.
* Non-parametric method: The decision tree is considered a non-parametric method. This means that decision trees have no assumptions        about the spatial distribution and structure of the classifier.
* Nonlinear relationships between parameters do not affect tree performance.
* The number of hyper parameters to adjust is almost nil. [5]

**Disadvantages**
* Overfitting: Students in the decision tree can create overly complex trees that do not generalize the data well. This is called overfitting. Overfitting is one of the most practical difficulties for decision tree models. This problem is solved by setting restrictions on the model and pruning parameters.
* Not suitable for continuous variables: When working with continuous numeric variables, the decision tree loses information when it classifies the variables into different categories.
* Decision trees can be unstable because small variations in the data can generate a completely different tree. This is called variance, which must be reduced by methods such as bagging and reinforcement.
* Greedy algorithms cannot guarantee the return of the globally optimal decision tree. This can be mitigated by training multiple trees, where features and samples were randomly sampled with replacement.
* Decision tree students create skewed trees if some classes dominate. Therefore, it is recommended to balance the dataset before adjusting it with the decision tree.
* Gaining information in a decision tree with categorical variables gives a biased response for attributes with higher no. of categories. [5]
* Overall, it offers low prediction accuracy for a data set compared to other machine learning algorithms.
* Calculations can become complex when there are many class labels.

**Logistic Regression**
Overfitting: Students in the decision tree can create overly complex trees that do not generalize the data well. This is called overfitting. Overfitting is one of the most practical difficulties for decision tree models. This problem is solved by setting restrictions on the model and pruning parameters.
Not suitable for continuous variables: When working with continuous numeric variables, the decision tree loses information when it classifies the variables into different categories.
Decision trees can be unstable because small variations in the data can generate a completely different tree. This is called variance, which must be reduced by methods such as bagging and reinforcement.
Greedy algorithms cannot guarantee the return of the globally optimal decision tree. This can be mitigated by training multiple trees, where features and samples were randomly sampled with replacement.
Decision tree students create skewed trees if some classes dominate. Therefore, it is recommended to balance the dataset before adjusting it with the decision tree.
Gaining information in a decision tree with categorical variables gives a biased response for attributes with higher no. of categories. [5]
Overall, it offers low prediction accuracy for a data set compared to other machine learning algorithms.
Calculations can become complex when there are many class labels.

**Adventages**
* It is a technique widely used by data scientists due to its effectiveness and simplicity.
* It is not necessary to have large computational resources, both in training and in execution.
* The results are highly interpretable.
* The weight of each of the characteristics determines the importance it has in the final decision.
* They use attributes related to the output. [one]

**Disadvantages**
* Impossibility of directly solving non-linear problems.
* Requires that the target variable is to be linearly separable. Otherwise, the logistic regression model will not classify correctly.
* Shows dependency on characteristics.
* It is not one of the most powerful algorithms that exist. [one]

**Multiplayer Perceptron**
It is an artificial neural network (RNA) formed by multiple layers, this allows you to solve problems that are not linearly separable, which is the main limitation of the perceptron (also called simple perceptron). The multilayer perceptron can be fully or locally connected.
This model consists of several layers of interconnected computing units; Each neuron in a layer is directly connected to the neurons in the previous layer. The model is based on functions since each unit of the mentioned networks applies an activation function. [2]

**Advantages**
* It allows its use to solve problems of pattern association, image segmentation, data compression, etc.
* Obtaining accurate estimates.
* You can train a network that predicts two values. [2]

***Disadvantages**
* They can only solve linearly separable problems.
* It does not extrapolate well, that is, if the network is poorly or poorly trained, the outputs may be imprecise.
* The existence of local minima in the error function considerably makes training difficult, since once a minimum has been reached,       training stops even if the set convergence rate has not been reached. [2]


<a name="chapter4"></a>
## CHAPTER IV.  Implementation

**Tools used**
All the tools used to obtain the final results are described, from software and their versions including the data set and the coding.

**Spark with scala**
The Apache Spark programming language was used with Scala for its ease of use, speed and API to operate large amounts of data, as well as being modern and object-oriented. Visual Code was also used as a source code editor.

**Bookstores**
The ml, Mlib and SQL libraries including the following were used: VectorAssembler, Vectors, StringIndexer, MultilayerPerceptronClassifier, MulticlassClassificationEvaluator, LogisticRegression, DecisionTree, MLUtils. As well as the Sql SparkSession library.

**Versions**
Spark version 2.4.5
Scala version 2.11.12
Visual Code version 1.46

**Dataset**
The dataset called “bank-full.csv” was used, of which the use of certain data was limited in each case.
The columns balance "," day "," duration "," pdays "," previous "were used to generate the characteristics prior to their transformation

**Coding**
Explain what libraries were needed each of the algorithms and the steps to obtain the result.

**SVM**
For the implementation of Support Vector Machine, the import of the libraries "LinearSVC", "MulticlassClassificationEvaluator" "VectorAssembler", "Transformer" and "SparkSession" was started, later the lines that allow reducing errors were included, a spark session was created and The file was loaded, the transformation of the data was made with "VectorAssembler" using the characteristics, then the change was made with the column "and" by "label" and both label and characteristics were combined in a table.
The data with the “split” function was divided into 70% for training and 30% for testing, an arrangement was created to carry out input and output tests, the model was adjusted and the model's accuracy was printed using the function evaluator "MultiClassClassificationEvaluator" and finally the time the process took was shown.

**Decision Tree**
For the implementation of Support Vector Machine, the import of the libraries “DecisionTreeClassificationModel”, “MulticlassClassificationEvaluator”, “VectorAssembler”, “Transformer” and “SparkSession” was started, later the lines that allow reducing errors were included, a spark session was created and The file was loaded, the transformation of the data was made with "VectorAssembler" using the characteristics, then the change was made with the column "and" by "label" and both label and characteristics were combined in a table.
The data with the “split” function was divided into 70% for training and 30% for testing, an arrangement was created to carry out input and output tests, the model was adjusted and the model's accuracy was printed using the function evaluator "MultiClassClassificationEvaluator" and finally the time the process took was shown.

**Logistic Regression.**
For the implementation of logistic regression, the import of the “LogisticRegression”, “VectorAssembler”, “Transformer” and “SparkSession” libraries began, later the lines that allow reducing errors were included, a spark session was created and the file was loaded , the transformation of the data was made with "VectorAssembler" using the characteristics, then the change was made with the column "and" by "label" and both label and characteristics were combined in a table.
The “LogisticRegression” object was applied with 10 iterations, the model was adjusted to print the coefficients and intersections as well as the precision and the time of the process.

**Multilayer Perceptron.**
For the implementation of Multilayer Perceptron, it started with the import of the libraries "MultilayerPerceptronClassifier", "MulticlassClassificationEvaluator" "VectorAssembler", "Transformer" and "SparkSession", later the lines that allow reducing errors were included, a spark session was created and loaded the file, the transformation of the data was made with "VectorAssembler" using the characteristics, then the change was made with the column "and" by "label" and both label and characteristics were combined in a table.
The data with the “split” function was divided into 60% for training and 40% for testing, an arrangement was created to carry out input and output tests, the model was adjusted and the model's accuracy was printed using the function evaluator "MultiClassClassificationEvaluator" and finally the time the process took was shown.

<a name="chapter5"></a>
## CHAPTER V.  Results
Note: For each iteration, precision and processing time were taken into account, taking a division structure (Accuracy / Time)

*Table 1. Comparative table of the performance of the SVM algorithm*

![Grafica](https://github.com/alejandrodominguezt/Datos_Masivos/blob/development/Table1.PNG?raw=true)

**Result**
The SVM model had 88% accuracy when tested in 10 iterations, the lowest value was obtained on lap number 8 while the most accurate value was shown on lap number 3.
The average process time was 9.5502, on lap 6 the longest processing time was given, while the first iteration was the fastest to be processed.

*Table 2. Comparative table of the performance of the Decision Tree algorithm.*
![Grafica](https://github.com/alejandrodominguezt/Datos_Masivos/blob/development/table2.PNG?raw=true)

**Result**
The Decision Trees model had 89% accuracy when tested in 10 iterations, the lowest value was obtained on round number 9 while the most accurate value was shown on turns number 2, 4 and 8.
The average process time was 4.4045. The last processing time was given in the last lap, while the first iteration was the fastest in being processed.

*Tabla 3. Tabla comparativa del rendimiento del algoritmo de Regresión Logística.*
![Grafica](https://github.com/alejandrodominguezt/Datos_Masivos/blob/development/Table3.PNG?raw=true)

**Result**
The Decision Trees model had 89.62% accuracy when tested in 10 iterations, the lowest value was obtained on rounds number 1 and 9 while the most accurate value was shown on rounds number 2, 4, 5 and 10. .
The average process time was 0.7181, the last processing time was given in the last lap, while the second iteration was the fastest to be processed.

*Table 4. Comparative table of the performance of the Multilayer Perceptron algorithm.*
![Grafica](https://github.com/alejandrodominguezt/Datos_Masivos/blob/development/table4.PNG?raw=true)

**Result**
The Multilayer Perceptron model had an 88.4% accuracy when tested in 10 iterations, this being the unique value in all iterations.
The average process time was 2.5283 in the second round, the longest processing time was given, while the ninth iteration was the fastest to be processed.

**Comparative results**
The following structure was used for the comparative results table:
Avg = Average
SVM = SVM
AD = Decision Tree
RL = Logistic Regression
PM = Multilayer Perceptron
Accu = Accuracy
Time = Time

![Grafica](https://github.com/alejandrodominguezt/Datos_Masivos/blob/development/table5.PNG?raw=true)

<a name="chapter6"></a>
## CHAPTER VI.  Conclusions
Based on the results obtained and analyzing the comparative tables, we consider that the 
Linear Regression model is the most accurate with 89.6% on average, being the closest to 
100%, this model was also observed to be the fastest in times of processing with an average 
of 0.7181 being the least compared to other models.
On the other hand, the slowest model with poor precision among those studied is SVM, as it
is 88% accurate with a processing time of 9.5 (the highest of all models).

<a name="referencces"></a>
## References
[1] Rodríguez, D. (2018). Logistic regression. June 13, 2020, from Analyticslane Website: https://www.analyticslane.com/2018/07/23/la-regresion-logistica/

[2] Jackeline, L. (2015). Multilayer Perceptron. June 13, 2020, from Blogsplot Website: http://jackelineliz.blogspot.com/

[3] Patel, S. (2018, November 10). Chapter 2: SVM (Support Vector Machine) - Theory - Machine Learning 101. Retrieved June 13, 2020, from https://medium.com/machine-learning-101/chapter-2-svm-support-vector-machine- theory-f0812effc72

[4] Support Vector Machines. (s. f.). Retrieved June 13, 2020, from https://scikit-learn.org/stable/modules/svm.html#support-vector-machines

[5] Brid, R. S. (2018, October 26). Decision Trees - A simple way to visualize a decision. Retrieved June 13, 2020, from https://medium.com/greyatom/decision-trees-a-simple-way-to-visualize-a-decision-dc506a403aeb#:%7E:text=A%20decision%20tree% 20is% 20a,% 2C% 20resource% 20costs% 2C% 20and% 20utility.


**Original Document**
https://docs.google.com/document/d/1rhGGaYyqrcg4jcVrEOeTYzCGuGAXWcv4hIiDPL7fOuw/edit?usp=sharing
