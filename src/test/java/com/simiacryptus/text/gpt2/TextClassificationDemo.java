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

package com.simiacryptus.text.gpt2;

import com.simiacryptus.lang.SerializableFunction;
import com.simiacryptus.lang.TimedResult;
import com.simiacryptus.lang.Tuple2;
import com.simiacryptus.mindseye.eval.ArrayTrainable;
import com.simiacryptus.mindseye.lang.Coordinate;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.SerialPrecision;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.java.EntropyLossLayer;
import com.simiacryptus.mindseye.layers.java.LinearActivationLayer;
import com.simiacryptus.mindseye.layers.java.SoftmaxLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.network.SimpleLossNetwork;
import com.simiacryptus.mindseye.opt.IterativeTrainer;
import com.simiacryptus.mindseye.test.NotebookReportBase;
import com.simiacryptus.text.ClassifyUtil;
import com.simiacryptus.text.ProjectorUtil;
import com.simiacryptus.text.TensorStats;
import org.apache.commons.io.FileUtils;
import org.jetbrains.annotations.NotNull;
import org.junit.Test;

import javax.annotation.Nonnull;
import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.*;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.zip.ZipFile;

public class TextClassificationDemo extends NotebookReportBase {

  public static final File base = new File("C:\\Users\\andre\\Downloads\\twitter-sentiment-analysis2");
  private final int[] featureDims = {1, 24, 2, 16, 1, 64};

  public static Coordinate[] mostSignifigantClassifierPins(List<String[]> rows, int primaryKey, int secondaryKey, int maxDims, int[] featureDims) {
    Map<Integer, Map<Integer, Tensor>> indexedData = toTensorMap(rows, featureDims);
    Map<Integer, TensorStats> statsMap = ClassifyUtil.mapValues(indexedData, entry -> TensorStats.create(entry.values()));
    return ClassifyUtil.mostSignifigantCoords(statsMap.get(primaryKey), statsMap.get(secondaryKey), maxDims);
  }

  public static Map<Integer, Map<Integer, Tensor>> toTensorMap(List<String[]> rows, int[] featureDims) {
    return rows.stream().collect(Collectors.groupingBy((String[] row) -> {
      return Integer.parseInt(row[0]);
    }, Collectors.toMap((String[] row) -> {
      return Integer.parseInt(row[1]);
    }, (String[] row) -> {
      return new Tensor(SerialPrecision.Float.parse(row[2]), featureDims);
    })));
  }

  @Nonnull
  @Override
  public ReportType getReportType() {
    return ReportType.Applications;
  }

  @Override
  protected Class<?> getTargetClass() {
    return GPT2Model.class;
  }

  @Test
  public void splitTrainingData() throws IOException {
    List<String> sourceLines = FileUtils.readLines(new File(base, "train.csv"), "UTF-8");
    String header = sourceLines.get(0);
    ArrayList<String> strings = new ArrayList<>(sourceLines.subList(1, sourceLines.size()));
    Collections.shuffle(strings);
    ArrayList<Object> training = new ArrayList<>();
    training.add(header);
    int splitAt = (int) (strings.size() * 0.9);
    training.addAll(strings.subList(0, splitAt));
    FileUtils.writeLines(new File(base, "train_use.csv"), training);
    ArrayList<Object> holdout = new ArrayList<>();
    holdout.add(header);
    holdout.addAll(strings.subList(splitAt, strings.size()));
    FileUtils.writeLines(new File(base, "holdout.csv"), holdout);
  }

  @Test
  public void buildIndex() throws IOException {
    File indexFile = new File(base, "index.csv");
    int currentlyIndexed = indexFile.exists() ? FileUtils.readLines(indexFile, "UTF-8").size() : 0;
    File file = new File(base, "train_use.csv");
    List<String[]> rows = Arrays.stream(FileUtils.readFileToString(file, "UTF-8").split("\n"))
        .map(s -> s.split(",", 3))
        .collect(Collectors.toList());
    @NotNull Function<String, Future<Tensor>> languageTransform = ClassifyUtil.getLanguageTransform(1);
    int batchSize = 10;
    for (int startRow = 1 + currentlyIndexed; startRow < rows.size(); startRow += batchSize) {
      logger.info("Processing rows from " + startRow);
      List<String[]> subList = rows.subList(startRow, Math.min(rows.size(), startRow + batchSize));
      TimedResult<Void> time = TimedResult.time(() -> {
        List<List<String>> dataTable = subList.stream().map(strs -> {
          try {
            return Arrays.asList(
                strs[1],
                strs[0],
                SerialPrecision.Float.base64(languageTransform.apply(strs[2]).get()),
                strs[2]
            );
          } catch (Exception e) {
            throw new RuntimeException(e);
          }
        }).collect(Collectors.toList());
        CharSequence data = dataTable.stream()
            .map(s -> s.stream().reduce((a, b) -> a + "," + b).orElse(""))
            .reduce((a, b) -> a + "\n" + b).orElse("");
        FileUtils.write(indexFile, data, "UTF-8", true);
      });
      logger.info("Wrote in " + time.seconds());
      System.gc();
    }

  }

  @Test
  public void projector() throws IOException, URISyntaxException {
    List<String[]> rows = loadIndexFile();

    Function<Tensor, Tensor> tensorFunction;
    {
      Coordinate[] coords = mostSignifigantClassifierPins(rows, 0, 1, 100, featureDims);
      tensorFunction = Tensor.select(coords);
    }

    Map<String, Tensor> tensorMap = rows.stream().collect(Collectors.toMap(r -> r[3], r -> tensorFunction.apply(new Tensor(SerialPrecision.Float.parse(r[2]), featureDims))));
    ProjectorUtil.browseProjector(ProjectorUtil.publishProjector(tensorMap));
  }

  @Test
  public void trainModel() throws IOException {
    List<String[]> rows = loadIndexFile();
    Map<Integer, Map<Integer, Tensor>> indexedData = toTensorMap(rows, featureDims);
    Tuple2<PipelineNetwork, SerializableFunction<Tensor, Tensor>> tuple2 = ClassifyUtil.buildClassifier(ClassifyUtil.mapValues(indexedData, entry -> new ArrayList<>(entry.values())));
    PipelineNetwork classfier = tuple2._1;
    List<Tensor[]> indexData = rows.stream().map(e -> {
      return new Tensor[]{
          new Tensor(2).set(Integer.parseInt(e[0]), 1),
          tuple2._2.apply(new Tensor(SerialPrecision.Float.parse(e[2]), featureDims))
      };
    }).collect(Collectors.toList());
    classfier.wrap(pretrain(classfier, indexData));
    classfier.wrap(new SoftmaxLayer());
    Tensor[][] trainingData = IntStream.range(0, indexData.size()).mapToObj(i -> new Tensor[]{
        indexData.get(i)[1],
        indexData.get(i)[0]
    }).toArray(i -> new Tensor[i][]);
    double trainingResult = IterativeTrainer.wrap(new ArrayTrainable(trainingData, new SimpleLossNetwork(classfier, new EntropyLossLayer()), 1000))
        .setMaxIterations(100)
        .setTimeout(5, TimeUnit.MINUTES)
        .runAndFree();
    logger.info(String.format("Training Result: %s", trainingResult));
    classfier.writeZip(new File(base, "model.zip"));
  }

  @Test
  public void testModel() throws IOException {
    Layer classifier = Layer.fromZip(new ZipFile(new File(base, "model.zip")));
    File file = new File(base, "holdout.csv");
    List<String[]> rows = Arrays.stream(FileUtils.readFileToString(file, "UTF-8").split("\n"))
        .map(s -> s.split(",", 3))
        //.limit(10)
        .collect(Collectors.toList());
    @NotNull Function<String, Future<Tensor>> languageTransform = ClassifyUtil.getLanguageTransform(1);
    int batchSize = 10;
    for (int startRow = 1; startRow < rows.size(); startRow += batchSize) {
      logger.info("Processing rows from " + startRow);
      List<String[]> subList = rows.subList(startRow, Math.min(rows.size(), startRow + batchSize));
      TimedResult<String> time = TimedResult.time(() -> {
        Tensor[][] evalData = subList.stream().map((String[] strs) -> {
          try {
            Tensor category = new Tensor(2).set(Integer.parseInt(strs[1]), 1);
            Tensor nlpTransform = languageTransform.apply(strs[2]).get();
            Tensor prediction = classifier.eval(nlpTransform).getDataAndFree().getAndFree(0);
            return new Tensor[]{
                category,
                nlpTransform,
                prediction
            };
          } catch (Throwable e) {
            throw new RuntimeException(e);
          }
        }).toArray(i -> new Tensor[i][]);

        EntropyLossLayer entropyLossLayer = new EntropyLossLayer();
        double totalEntropy = Arrays.stream(evalData).mapToDouble(row -> {
          return entropyLossLayer.eval(row[2], row[0]).getDataAndFree().getAndFree(0).get(0);
        }).sum();
        double accuracy = Arrays.stream(evalData).mapToDouble(row -> {
          Tensor prediction = row[2];
          Tensor category = row[0];
          int index = prediction.coordStream(false).sorted(Comparator.comparing(c -> -prediction.get(c))).findFirst().get().getIndex();
          return category.get(index);
        }).average().getAsDouble();
        return String.format("accuracy=%s; entropy=%s", accuracy, totalEntropy);
      });
      logger.info(String.format("Processed rows %s-%s in %s: %s", startRow, startRow + subList.size(), time.seconds(), time.result));
      System.gc();
    }
  }

  @NotNull
  protected LinearActivationLayer pretrain(PipelineNetwork classfier, List<Tensor[]> indexData) {
    LinearActivationLayer activationLayer = new LinearActivationLayer().setScale(-1e-9).setBias(0);
    List<Tensor> preEval = classfier.map(indexData.stream().map(x -> x[1]).collect(Collectors.toList()));
    Tensor[][] trainingData = IntStream.range(0, indexData.size()).mapToObj(i -> new Tensor[]{
        preEval.get(i),
        indexData.get(i)[0]
    }).toArray(i -> new Tensor[i][]);
    double trainingResult = IterativeTrainer.wrap(new ArrayTrainable(trainingData, new SimpleLossNetwork(PipelineNetwork.wrap(1,
        activationLayer.addRef(),
        new SoftmaxLayer()
    ), new EntropyLossLayer())))
        .setMaxIterations(100)
        .setTimeout(5, TimeUnit.MINUTES)
        .runAndFree();
    logger.info(String.format("Training Result: %s", trainingResult));
    return activationLayer;
  }

  protected List<String[]> loadIndexFile() throws IOException {
    File file = new File(base, "index.csv");
    return FileUtils.readLines(file, "UTF-8").stream()
        .map(s -> s.split(",", 4)).collect(Collectors.toList());
  }

}


