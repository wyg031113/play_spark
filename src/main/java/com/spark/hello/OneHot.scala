package com.spark.hello

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.feature.{OneHotEncoder, OneHotEncoderEstimator, OneHotEncoderModel, StringIndexer, VectorAssembler}
import org.apache.spark.sql.catalyst.dsl.expressions.StringToAttributeConversionHelper
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{DataFrame, SQLContext, SparkSession}
import org.apache.spark.{SparkConf, SparkContext, sql}

import scala.collection.mutable.ListBuffer

object OneHot {
    def main(args: Array[String]): Unit = {
        //Logger.getLogger("org").setLevel(Level.ERROR)
        oneHotTest()
    }
    def oneHotMovie(): Unit ={
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
        oneHotEncoder(samples)
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
}
