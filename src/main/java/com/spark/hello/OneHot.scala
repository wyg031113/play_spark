package com.spark.hello

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.feature.{MinMaxScaler, OneHotEncoder, OneHotEncoderEstimator, OneHotEncoderModel, QuantileDiscretizer, StringIndexer, VectorAssembler}
import org.apache.spark.sql.catalyst.dsl.expressions.StringToAttributeConversionHelper
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{DataFrame, SQLContext, SparkSession}
import org.apache.spark.{SparkConf, SparkContext, sql}
import org.apache.spark.sql.functions._
import org.spark_project.dmg.pmml.Quantile

import scala.collection.mutable.ListBuffer
//参考https://blog.csdn.net/u011622631/article/details/81562699
object OneHot {
    def main(args: Array[String]): Unit = {
        //Logger.getLogger("org").setLevel(Level.ERROR)
        //oneHotTest()
        //HotTest()
        numFeature()
    }
    def HotTest(): Unit ={
        val conf = new SparkConf()
            .setAppName("OneHot")
            .setMaster("local")
        val spark = SparkSession.builder().config(conf).getOrCreate()
        val moviePath = this.getClass.getResource("/data/movies.csv")
        val samples = spark.read.format("csv")
            .option("header", "true")
            .load(moviePath.getPath)
        samples.printSchema()
        samples.show()
        //oneHotEncoder(samples)
        multiHot(samples)
        spark.close()
    }
    def oneHotEncoder(samples :DataFrame): Unit = {
        val sampleIdInt = samples.withColumn("movieIdNumber", col("movieId").cast(sql.types.IntegerType))
        val encoder = new OneHotEncoderEstimator()
            .setInputCols(Array("movieIdNumber"))
            .setOutputCols(Array("movieIdVector"))
            .setDropLast(false)
        val oneHotSamples = encoder.fit(sampleIdInt)
            .transform(sampleIdInt)
        oneHotSamples.printSchema()
        oneHotSamples.show(10)
    }
    def oneHotTest(): Unit ={
        //初始化环境
        val conf = new SparkConf()
            .setMaster("local[4]")
            .setAppName(getClass.getSimpleName)
            .set("spark.testing.memory", "1000000000")
        val spark = SparkSession.builder()
            .config(conf)
            .getOrCreate()
        //读取数据
        val sample = spark.read.json(this.getClass.getResource("/data/lr_test03.json").getPath)
        sample.show()
        //类别型的列
        val cateCol = Array("gender", "children")
        //stage pipeline来处理数据
        val stages = new ListBuffer[PipelineStage]()
        for(cate <- cateCol){
            //能把输入中的类别按出现频率从小到达转换成数字
            val indexer = new StringIndexer()
                .setInputCol(cate)
                .setOutputCol(s"${cate}Index")
            // one hot编码
            val encoder =  new OneHotEncoderEstimator()
                .setInputCols(Array(indexer.getOutputCol))
                .setOutputCols(Array(s"${cate}ClassVec"))
            stages.append(indexer, encoder)
        }
        //数值型的列
        val numCols = Array("affairs", "age", "yearsmarried",
                "religiousness", "education", "occupation", "rating")
        //连接数值型的列和类别型的列
        val assemblerInputs = cateCol.map(_+"ClassVec") ++ numCols
        val assembler = new VectorAssembler()
            .setInputCols(assemblerInputs).setOutputCol("feature")
        stages.append(assembler)
        //执行Pipeline
        val pipeline = new Pipeline()
        pipeline.setStages(stages.toArray)
        val pipelineModel = pipeline.fit(sample)
        val dataset = pipelineModel.transform(sample)
        dataset.show()

        //分为训练集合，测试集
        val Array(trainDF, testDF) = dataset.randomSplit(Array(0.6, 0.4), seed = 12345)
        println(s"train size:${trainDF.count()}, test size: ${testDF.count()}")
        //逻辑回归训练模型
        var lrModel = new LogisticRegression()
            .setLabelCol("affairs")
            .setFeaturesCol("feature")
            .fit(trainDF)
        //在测试集上预测
        var predictions = lrModel.transform(testDF)
        println("predictions....")
        predictions.show(10)
        //评估准确性
        println("accuracy....")
        val preds = predictions.withColumnRenamed("affairs", "label")
        val evaluator = new BinaryClassificationEvaluator()
        evaluator.setMetricName("areaUnderROC")
        val auc = evaluator.evaluate(preds)
        println(s"areaUnderRoc=${auc}")
    }
    def multiHot(sample :DataFrame): Unit ={
        //genres分割
        val samplesWithGenre = sample.select(col("movieId"), col("title"),
            explode(split(col("genres"), "\\|").cast("array<string>")).as("genre"))
        //编码，每个genre用一个数字表示
        val strIndexer = new StringIndexer()
            .setInputCol("genre")
            .setOutputCol("genreIndex")
        val strIndexModel = strIndexer.fit(samplesWithGenre)
        val genreIndexSamples = strIndexModel.transform(samplesWithGenre)
            .withColumn("genreIndexInt", col("genreIndex").cast(sql.types.IntegerType))
        genreIndexSamples.show()
        //有多少种genre
        val indexSize = genreIndexSamples
            .agg(max(col("genreIndexInt")))
            .head()
            .getAs[Int](0) + 1
        println(s"max index is ${indexSize}")

        //反向explode
        val processSamples =genreIndexSamples
            .groupBy(col("movieId"))
            .agg(collect_list("genreIndexInt").as("genreIndexes"))
            .withColumn("indexSize", typedLit(indexSize)) //加上常数列
        processSamples.show()
        //UDF, udf参数必须是column,所以typedLit
        val array2vec: UserDefinedFunction = udf {
            (a: Seq[Int], length: Int) =>
                org.apache.spark.ml.linalg.Vectors.sparse(length, a.sortWith(_ < _).toArray, Array.fill[Double](a.length)(1.0)) }
        val finalSample = processSamples
            .withColumn("vector", array2vec(col("genreIndexes"), col("indexSize")))
        finalSample.show()
    }
    def numFeature(): Unit ={
        val conf = new SparkConf()
            .setAppName(this.getClass.getSimpleName)
            .setMaster("local")
        val spark = SparkSession.builder()
            .config(conf)
            .getOrCreate()
        val moviesFile = this.getClass.getResource("/data/ratings.csv")
        val movies = spark.read.format("csv")
            .option("header", "true")
            .load(moviesFile.getPath)
        movies.printSchema()
        movies.show()

        val moviesNumCol = movies.withColumn("userIdInt", col("userId").cast(org.apache.spark.sql.types.IntegerType))
            .withColumn("movieIdInt", col("movieId").cast(org.apache.spark.sql.types.IntegerType))
            .withColumn("ratingDouble", col("rating").cast(org.apache.spark.sql.types.DoubleType))
        moviesNumCol.printSchema()
        moviesNumCol.show()

        val encoder = new OneHotEncoderEstimator()
            .setInputCols(Array("userIdInt", "movieIdInt"))
            .setOutputCols(Array("userIdVec", "movieIdVec"))
            .setDropLast(false)
        val afterOnehot = encoder.fit(moviesNumCol).transform(moviesNumCol)
        afterOnehot.show()
        //上边对两列进行oneHot,纯属娱乐

        val double2vec: UserDefinedFunction = udf { (value: Double) => org.apache.spark.ml.linalg.Vectors.dense(value) }

        val moviesFeature = movies
            .groupBy(col("movieId"))
            .agg(count(lit(1)).as("ratingCount"),
                avg(col("rating")).as("avgRating"),
                variance(col("rating")).as("varRating"))
            .withColumn("avgRatingVec", double2vec(col("avgRating")))
        moviesFeature.printSchema()
        moviesFeature.show(10)
        //分桶
        val ratingCountDis = new QuantileDiscretizer()
            .setInputCol("ratingCount")
            .setOutputCol("ratingCountBucket")
            .setNumBuckets(100)
        //normalization
        val ratingScalar = new MinMaxScaler()
            .setInputCol("avgRatingVec")
            .setOutputCol("scaleAvgRating")

        val pipelineStage = Array[PipelineStage](ratingCountDis, ratingScalar)
        val featurePipeline = new Pipeline().setStages(pipelineStage)
        val movieProcessFeature = featurePipeline.fit(moviesFeature).transform(moviesFeature)
        movieProcessFeature.show()

    }
}

