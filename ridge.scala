import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.feature.Normalizer
import org.apache.spark.mllib.regression.RidgeRegressionWithSGD
val data = sc.textFile("lpsa.dat")

val parsedData = data.map { line =>
val x : Array[String] = line.replace(",", " ").split(" ")
val y = x.map{ (a => a.toDouble)}
val d = y.size - 1
val c = Vectors.dense(y(1),y(d))
LabeledPoint(y(0), c)
}.cache()


val Array(trainingData, testData) = df.randomSplit(Array(0.8, 0.2))

val numIterations = 100
val learningRate = 0.00000001
val alpha = 0.01
/**
* This method uses stochastic gradient descent (SGD) algorithm to train a ridge regression model
*
* @param trainingDataset           Training dataset as a JavaRDD of LabeledPoints
* @param noOfIterations            Number of iterarations
* @param initialLearningRate       Initial learning rate (SGD step size)
* @param regularizationParameter   Regularization parameter
* @param miniBatchFraction         SGD minibatch fraction
* @return                          Ridge regression model
*/
val model = RidgeRegressionWithSGD.train(trainingData, numIterations, learningRate, alpha)

val valuesAndPreds = trainingData.map { point =>
val prediction = model.predict(point.features)
(point.label, prediction)
}

valuesAndPreds.foreach((result) => println(s"predicted label: ${result._1}, actual label: ${result._2}"))

val MSE = valuesAndPreds.map{ case(v, p) => math.pow((v - p), 2) }.mean()
println("training Mean Squared Error = " + MSE)
