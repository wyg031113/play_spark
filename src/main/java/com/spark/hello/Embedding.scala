package com.spark.hello

import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.{Word2Vec, Word2VecModel}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{array_contains, array_join, col, collect_list, struct, udf}
import org.apache.spark.sql.types.{StringType, StructField, StructType}

import java.io.{BufferedWriter, File, FileWriter}
import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scala.util.Random
import scala.util.control.Breaks.{break, breakable}


object Embedding {
    def main(args: Array[String]): Unit = {
        val conf = new SparkConf()
            .setMaster("local")
            .setAppName(this.getClass.getSimpleName)
        val spark = SparkSession.builder().config(conf).getOrCreate()
        val movieRatingFile = this.getClass.getResource("/data/ratings.csv")
        val userSeq = prepareSequenceData(spark, movieRatingFile.getPath)
        trainItem2Vec(spark, userSeq, 10, "item2vecEmb.csv")
        val trans = graphEmbMatrix(spark, userSeq)
        val sampleWalk = randomWalk(trans._1, trans._2)
        val rddSamples = spark.sparkContext.parallelize(sampleWalk)
        import spark.implicits._
        val sampleDF = rddSamples.toDF("movieIdStr")
        val model = trainItem2Vec(spark, sampleDF , 10, "item2vecGraphEmb.csv")
    }

    def prepareSequenceData(spark :SparkSession, dataPath :String): DataFrame ={

        val movieRating = spark.read.format("csv")
            .option("header", "true")
            .load(dataPath)
        movieRating.printSchema()
        movieRating.show()
        val sortUdf :UserDefinedFunction = udf((rows :Seq[Row])=>{
            rows.map{
                case Row(movieId:String, timestamp: String)=>(movieId, timestamp)
            }
                .sortBy{
                    case(_, timestamp) => timestamp
                }
                .map{
                    case(movieId, _) => movieId
                }

        })
        val userSeq = movieRating
            .where(col("rating") >= 3.5)
            .groupBy("userId")
            .agg(sortUdf(collect_list(struct("movieId", "timestamp"))) as("movieIds"))
            .withColumn("movieIdStr", array_join(col("movieIds"), " "))
        userSeq.show()
        userSeq.select("userId", "movieIdStr").show(20, false)
        val str2seq: UserDefinedFunction = udf { (value: String) => value.split(" ").toSeq }

        userSeq.select("movieIdStr").withColumn("movieIdStr", str2seq(col("movieIdStr")))
       // userSeq.select("movieIdStr").rdd.map(r=>r.getAs[String](0).split(" ").toSeq)
    }
    def trainItem2Vec(spark :SparkSession, samples :DataFrame, embLength :Int, outputFileName :String): Unit = {
        val word2vec = new Word2Vec()
            .setVectorSize(embLength)
            .setWindowSize(5)
            .setNumPartitions(10)
            .setInputCol("movieIdStr")
        val model = word2vec.fit(samples)
        val sym = model.findSynonyms("592", 20)
        for(Row(s,c) <- sym.rdd){
            println(s"592-> ${s} ${c}")
        }
        val embFolder = this.getClass.getResource("/data/")
        val file = new File((embFolder.getPath + outputFileName))
        val bw = new BufferedWriter(new FileWriter(file))
        val samplesTrans = model.getVectors
        samplesTrans.printSchema()
        samplesTrans.show(30, false)
        samplesTrans.rdd.map{
            case Row(word, vec) => {
                val str = vec.toString
                str.substring(1, str.length-1)
            }
        }
        for(Row(k,v) <- model.findSynonyms("592", 10)) {
            println(s"${k} ${v}")
        }
        return model
    }

    def graphEmbMatrix(spark :SparkSession, samples :DataFrame): (mutable.HashMap[String, mutable.HashMap[String, Double]], mutable.HashMap[String, Double]) = {
        samples.printSchema()
        samples.show()
        val pairs = samples.rdd.flatMap{
            case Row(str:mutable.WrappedArray[String])=> {
                val ids = str
                val lst = new ListBuffer[(String, String)]()
                for (i <- 1 until ids.length) {
                    lst.append((ids(i - 1), ids(i)))
                }
                lst.toList
            }
        }
        val transMatrix = new mutable.HashMap[String, mutable.HashMap[String, Double]]()
        val vexOutNum = new mutable.HashMap[String, Double]()
        val cntPairs = pairs.countByValue()
        var totalCnt :Long= 0
        cntPairs.foreach(v =>{
            val pair = v._1
            val cnt = v._2
            if(!transMatrix.contains(pair._1)) {
                transMatrix(pair._1) = mutable.HashMap[String, Double]()
            }
            transMatrix(pair._1)(pair._2) = v._2
            vexOutNum(pair._1) = vexOutNum.getOrElse[Double](pair._1, 0) + cnt
            totalCnt = totalCnt + cnt
        })
        for(x <- transMatrix) {
            for(y <- x._2) {
                transMatrix(x._1)(y._1) /= vexOutNum(x._1)
            }
        }
        for(x <- vexOutNum) {
            vexOutNum(x._1) /= totalCnt
        }

        return (transMatrix, vexOutNum)


    }
    def randomWalk(transMatrix :mutable.HashMap[String, mutable.HashMap[String, Double]],
                   vexDistribute :mutable.HashMap[String, Double]):Seq[Seq[String]] = {
        val sampleCount = 20000
        val sampleLength = 10
        val samples = mutable.ListBuffer[Seq[String]]()
        for(_ <- 1 to sampleCount) {
            val one = oneRandomWalk(transMatrix, vexDistribute, sampleLength)
            println(one)
            samples.append(one)
        }
        Seq(samples.toList:_*)
    }
    def oneRandomWalk(transMatrix :mutable.HashMap[String, mutable.HashMap[String, Double]],
                      vexDistribute :mutable.HashMap[String, Double],
                      sampleLength :Int) :Seq[String] = {
        var first = ""
        val sample = ListBuffer[String]()
        var prob :Double = 0
        var target :Double = Random.nextDouble()
        breakable{
            for(e <- vexDistribute){
                prob += e._2
                if(prob >= target){
                    first = e._1
                    break
                }

            }
        }
        sample.append(first)
        breakable{
            while(sample.length < sampleLength) {
                val start = sample(sample.length - 1)
                var next = ""
                target = Random.nextDouble()
                prob = 0
                if(!transMatrix.contains(start)){
                    break
                }
                val targetMap = transMatrix(start)
                breakable{
                    for (item <- targetMap) {
                        prob += item._2
                        if(prob >= target){
                            next = item._1
                            break
                        }

                    }
                }

                if (next == "") {
                    break
                }
                sample.append(next)
            }
        }
        return Seq(sample.toList:_*)
    }
}
