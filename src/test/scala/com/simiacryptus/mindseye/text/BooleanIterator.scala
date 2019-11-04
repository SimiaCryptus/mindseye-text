/*
 * Copyright (c) 2019 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.text

import java.net.InetAddress
import java.util
import java.util.concurrent.TimeUnit
import java.util.{UUID, function}

import com.fasterxml.jackson.annotation.JsonIgnore
import com.simiacryptus.aws.exe.EC2NodeSettings
import com.simiacryptus.lang.SerializableFunction
import com.simiacryptus.mindseye.art._
import com.simiacryptus.mindseye.art.util.ArtUtil._
import com.simiacryptus.mindseye.art.util._
import com.simiacryptus.mindseye.lang.cudnn.Precision
import com.simiacryptus.mindseye.lang.{SerialPrecision, Tensor}
import com.simiacryptus.mindseye.layers.cudnn.BandAvgReducerLayer
import com.simiacryptus.mindseye.network.PipelineNetwork
import com.simiacryptus.mindseye.text.BooleanIterator.{dim, featureDims}
import com.simiacryptus.notebook.{FormQuery, MarkdownNotebookOutput, NotebookOutput}
import com.simiacryptus.sparkbook.util.Java8Util._
import com.simiacryptus.sparkbook.util.LocalRunner
import com.simiacryptus.sparkbook.{AWSNotebookRunner, EC2Runner, NotebookRunner}
import com.simiacryptus.text.ClassifyUtil
import org.apache.spark.sql._
import org.apache.spark.sql.types._

import scala.collection.JavaConversions._
import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer
import scala.util.{Random, Try}

object TextClassifyUtil {
  def getSampleEntropy(classifierNetwork: PipelineNetwork, dimensionSelectionFunction: function.Function[Tensor, Tensor], row: Row): Double = {
    val tensor = new Tensor(SerialPrecision.Float.parse(row.getAs[String]("tensorSrc")), featureDims: _*)
    val result = classifierNetwork.eval(dimensionSelectionFunction.apply(tensor)).getDataAndFree.getAndFree(0)
    val v = result.get(0)
    tensor.freeRef()
    result.freeRef()
    -v
  }
}

import com.simiacryptus.mindseye.text.TextClassifyUtil._

object BooleanIterator_EC2 extends BooleanIterator with EC2Runner[Object] with AWSNotebookRunner[Object] {

  override def urlBase: String = String.format("http://%s:1080/etc/", InetAddress.getLocalHost.getHostAddress)

  override def inputTimeoutSeconds = 600

  override def maxHeap = Option("55g")

  override def nodeSettings = EC2NodeSettings.P2_XL

  override def javaProperties: Map[String, String] = Map(
    "spark.master" -> spark_master
  )

  override def spark_master = "local[1]"

}

object BooleanIterator_Local extends BooleanIterator with LocalRunner[Object] with NotebookRunner[Object] {
  override val urlBase: String = "http://localhost:1080/etc/"

  override def http_port: Int = 1081

  override def inputTimeoutSeconds = 30

  override def spark_master = "local[1]"

  def s3bucket = "www.tigglegickle.com"

}

object BooleanIterator {
  val featureDims = Array(1, 24, 2, 16, 1, 64)
  val dim = Tensor.length(featureDims: _*)
  val csvschema = StructType(Array(
    StructField("sentiment", IntegerType),
    StructField("id", IntegerType),
    StructField("tensorSrc", StringType),
    StructField("text", StringType)
  ))
  var precision = Precision.Double

  def indexImages(visionPipeline: => VisionPipeline[VisionPipelineLayer], toIndex: Int, indexResolution: Int, archiveUrl: String)
                 (files: String*)
                 (implicit log: NotebookOutput, sparkSession: SparkSession): DataFrame = {
    val indexed = Try {
      val previousIndex = sparkSession.read.parquet(archiveUrl)
      previousIndex.select("id").rdd.map(_.getString(0)).distinct().collect().toSet
    }.getOrElse(Set.empty)
    val allFiles = Random.shuffle(files.filter(!indexed.contains(_)).toList).distinct.take(toIndex)
    if (allFiles.isEmpty) sparkSession.emptyDataFrame
    else index(visionPipeline, indexResolution, allFiles: _*)
  }

  def index(pipeline: => VisionPipeline[VisionPipelineLayer], imageSize: Int, images: String*)
           (implicit log: NotebookOutput, sparkSession: SparkSession) = {
    val rows = sparkSession.sparkContext.parallelize(images, images.length).flatMap(file => {
      val layers = pipeline.getLayers
      val canvas = Tensor.fromRGB(ImageArtUtil.load(log, file, imageSize))
      val tuples = layers.keys.foldLeft(List(canvas))((input, layer) => {
        val l = layer.getLayer
        val tensors = input ++ List(l.eval(input.last).getDataAndFree.getAndFree(0))
        l.freeRef()
        tensors
      })
      tuples.head.freeRef()
      val reducerLayer = new BandAvgReducerLayer()
      val rows = (layers.keys.map(_.name()) zip tuples.tail).toMap
        .mapValues(data => {
          val tensor = reducerLayer.eval(data).getDataAndFree.getAndFree(0)
          data.freeRef()
          val doubles = tensor.getData.clone()
          tensor.freeRef()
          doubles
        }).map(t => {
        Array(file, imageSize, pipeline.name, t._1, t._2)
      })
      reducerLayer.freeRef()
      println(s"Indexed $file")
      rows
    }).cache()
    sparkSession.createDataFrame(rows.map(Row(_: _*)), csvschema)
  }


}

import com.simiacryptus.mindseye.text.BooleanIterator._

abstract class BooleanIterator extends ArtSetup[Object] with BasicOptimizer {

  val positiveSeeds: Array[String] = "".split("\n").toList.toArray
  val negativeSeeds: Array[String] = "".split("\n").toList.toArray
  val initialSamples = 50
  val incrementalSamples = 50
  val sampleEpochs = 5
  val hiddenLayer1 = 128
  val dropoutSamples = 5
  val dropoutFactor = Math.pow(0.5, 0.5)
  val classificationPaintingBias = 0.5
  val signalMatchBias = 0.1

  @JsonIgnore def spark_master: String

  def urlBase: String

  override def cudaLog = false

  override def postConfigure(log: NotebookOutput) = {
    implicit val sparkSession = sparkFactory
    implicit val exeCtx = scala.concurrent.ExecutionContext.Implicits.global
    val index = sparkSession.read.schema(csvschema).csv("file:///C:/Users/andre/Downloads/twitter-sentiment-analysis2/index.csv").cache()
    index.printSchema()

    def findRows(example: Int*): Dataset[Row] = {
      index.where(index("id").isin(example: _*))
    }

    val positiveExamples = new ArrayBuffer[Row]()
    val negativeExamples = new ArrayBuffer[Row]()

    require(!index.isEmpty)
    var classifierNetwork: PipelineNetwork = null
    var dimensionSelectionFunction: SerializableFunction[Tensor, Tensor] = null

    def bestSamples(sample: Int) = sparkSession.createDataFrame(sparkSession.sparkContext.parallelize(index.rdd.sortBy(row => {
      getSampleEntropy(classifierNetwork, dimensionSelectionFunction, row)
    }).take(sample).toList), index.schema)

    def newConfirmationBatch(index: DataFrame, sample: Int, log: NotebookOutput) = {
      val text = index.select("id", "text").distinct().collect().map(r => r.getInt(0) -> r.getString(1)).toMap
      val seed = index.rdd.map(_.getAs[Int]("id")).distinct().take(sample).map(_ -> true).toMap
      val ids = seed.mapValues(_ => UUID.randomUUID().toString).toArray.toMap
      new FormQuery[Map[String, Boolean]](log.asInstanceOf[MarkdownNotebookOutput]) {

        override def height(): Int = 800

        override protected def getDisplayHtml: String = ""

        override protected def getFormInnerHtml: String = {
          (for ((k, v) <- getValue) yield {
            s"""<input type="checkbox" name="${ids(k.toInt)}" value="true">${text(k.toInt)}<br/>"""
          }).mkString("\n")
        }

        override def valueFromParams(parms: java.util.Map[String, String], files: java.util.Map[String, String]): Map[String, Boolean] = {
          (for ((k, v) <- getValue) yield {
            k -> parms.getOrDefault(ids(k.toInt), "false").toBoolean
          })
        }
      }.setValue(seed.map(t => t._1.toString -> t._2)).print().get(6000, TimeUnit.SECONDS)
    }

    def trainEpoch(log: NotebookOutput) = {
      withTrainingMonitor(monitor => {
        val classifierTuple = ClassifyUtil.buildClassifier(Map(
          0.asInstanceOf[Integer] -> negativeExamples.map(x => new Tensor(SerialPrecision.Float.parse(x.getAs[String]("tensorSrc")), 1, 1, dim)).toList.asJava,
          1.asInstanceOf[Integer] -> positiveExamples.map(x => new Tensor(SerialPrecision.Float.parse(x.getAs[String]("tensorSrc")), 1, 1, dim)).toList.asJava
        ).asJava)
        classifierNetwork = classifierTuple.getFirst
        dimensionSelectionFunction = classifierTuple.getSecond

        null
      })(log)
    }

    if (positiveExamples.isEmpty || negativeExamples.isEmpty) {
      // Build Tag Model
      val (newPositives, newNegatives) = newConfirmationBatch(index = index, sample = initialSamples, log = log).map(x => (x._1.toInt, x._2)).partition(_._2)
      positiveExamples ++= newPositives.keys.map(findRows(_).head()).toList
      negativeExamples ++= newNegatives.keys.map(findRows(_).head()).toList
    }
    trainEpoch(log = log)

    for (i <- 0 until sampleEpochs) {
      val (newPositives, newNegatives) = newConfirmationBatch(index = bestSamples(incrementalSamples), sample = initialSamples, log = log).map(x => (x._1.toInt, x._2)).partition(_._2)
      positiveExamples ++= newPositives.keys.map(findRows(_).head()).toList
      negativeExamples ++= newNegatives.keys.map(findRows(_).head()).toList
      trainEpoch(log = log)
    }

    null
  }

  @JsonIgnore def sparkFactory: SparkSession = {
    val builder = SparkSession.builder()
    import scala.collection.JavaConverters._
    ImageArtUtil.getHadoopConfig().asScala.foreach(t => builder.config(t.getKey, t.getValue))
    builder.master("local[8]").getOrCreate()
  }

  def stats(dataframe: Dataset[Row]) = {
    val pixels = dataframe.select("tensorSrc").rdd.map(x => SerialPrecision.Float.parse(x.getAs[String](0)))
    val dim = pixels.first().length
    val pixelCnt = pixels.count().toInt
    new Tensor(pixels.reduce((a, b) => a.zip(b).map(t => t._1 + t._2)).map(_ / pixelCnt), 1, 1, dim)
  }

  def select(index: DataFrame, exampleRow: Row, window: Int)(implicit sparkSession: SparkSession): DataFrame = {
    val files = index.rdd.sortBy(r => {
      val doublesA = SerialPrecision.Float.parse(r.getAs[String]("tensorSrc"))
      val doublesB = SerialPrecision.Float.parse(exampleRow.getAs[String]("tensorSrc"))
      doublesA.zip(doublesB)
        .map(t => t._1 - t._2).map(x => x * x).sum
    }).map(_.getAs[String]("id")).distinct.take(window).toSet
    index.filter(r => files.contains(r.getAs[String]("file")))
  }
}

