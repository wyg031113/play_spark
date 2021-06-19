package com.spark.hello

import org.apache.log4j.{Level, Logger}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object WordCount {
    def main(args: Array[String]): Unit = {
        //Logger.getLogger("org").setLevel(Level.ERROR)
        val conf = new SparkConf()
            .setMaster("local")
            .setAppName("WordCount")
            .set("spark.submit.deployMode", "client")
        val sc = new SparkContext(conf)
        val lines : RDD[String] = sc.textFile("data");
        val words : RDD[String] = lines.flatMap(_.split("[,| ]"))
        val wordTuple = words.map((_, 1))

        val wordToCount = wordTuple.reduceByKey(_+_)
        val arr = wordToCount.collect()
        arr.foreach(println)

        sc.stop()
    }
}
