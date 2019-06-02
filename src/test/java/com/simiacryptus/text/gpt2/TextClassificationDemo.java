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

import com.simiacryptus.lang.TimedResult;
import com.simiacryptus.mindseye.eval.ArrayTrainable;
import com.simiacryptus.mindseye.lang.Coordinate;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.SerialPrecision;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.tensorflow.TFIO;
import com.simiacryptus.mindseye.layers.java.*;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.network.SimpleLossNetwork;
import com.simiacryptus.mindseye.opt.IterativeTrainer;
import com.simiacryptus.mindseye.test.NotebookReportBase;
import com.simiacryptus.mindseye.test.TestUtil;
import org.apache.commons.io.FileUtils;
import org.jetbrains.annotations.NotNull;
import org.junit.Test;

import javax.annotation.Nonnull;
import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.zip.ZipFile;

public class TextClassificationDemo extends NotebookReportBase {

  private final int[] featureDims = {1, 24, 2, 16, 1, 64};
  private final File base = new File("C:\\Users\\andre\\Downloads\\twitter-sentiment-analysis2");

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
    int currentlyIndexed = indexFile.exists()?FileUtils.readLines(indexFile, "UTF-8").size():0;
    File file = new File(base, "train_use.csv");
    List<String[]> rows = Arrays.stream(FileUtils.readFileToString(file, "UTF-8").split("\n"))
        .map(s -> s.split(",", 3))
        //.limit(10)
        .collect(Collectors.toList());
    @NotNull Function<String, Future<Tensor>> languageTransform = getLanguageTransform(1);
    int batchSize = 10;
    for (int startRow = 1+currentlyIndexed; startRow < rows.size(); startRow += batchSize) {
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

  @NotNull
  protected Function<String, Future<Tensor>> getLanguageTransform(int threads) {
    ExecutorService pool = Executors.newFixedThreadPool(threads);
    ThreadLocal<GPT2Model> gpt2ModelThreadLocal = new ThreadLocal<GPT2Model>() {
      @Override
      protected GPT2Model initialValue() {
        return GPT2Util.getModel_345M();
      }
    };
    ThreadLocal<GPT2Codec> gpt2CodecThreadLocal = new ThreadLocal<GPT2Codec>() {
      @Override
      protected GPT2Codec initialValue() {
        return GPT2Util.getCodec_345M();
      }
    };
    return (String row) -> pool.submit(() -> {
      GPT2Model gpt2 = gpt2ModelThreadLocal.get();
      GPT2Codec codec = gpt2CodecThreadLocal.get();
      LanguageCodeModel copy = gpt2.copy();
      double[] entropies = codec.encode(row).stream().mapToDouble(code -> {
        float[] prediction = copy.eval(code);
        return IntStream.range(0, prediction.length).mapToDouble(i -> prediction[i])
            .map(x -> x * Math.log(x)).sum() / Math.log(2);
      }).toArray();
      Tensor state = TFIO.getTensor(copy.state());
      int slice = state.getDimensions()[4] - 1;
      return new Tensor(1, 24, 2, 16, 1, 64).setByCoord(c -> {
        int[] coords = c.getCoords();
        coords[4] = slice;
        return state.get(coords);
      });
    });
  }

  @Test
  public void projector() throws IOException, URISyntaxException {
    List<String[]> rows = loadIndexFile();
    Map<String, Tensor> tensorMap = rows.stream().collect(Collectors.toMap(r -> r[3], r -> new Tensor(SerialPrecision.Float.parse(r[2]), featureDims)));

    List<Tensor> values = new ArrayList<>(tensorMap.values());
    Tensor avg = TestUtil.avg(values);
    logger.debug("Avg: " + avg.prettyPrint());
    BiasLayer biasLayer = new BiasLayer(avg.getDimensions()).set(avg.scaleInPlace(-1));
    Tensor scales = TestUtil.sum(
        PipelineNetwork.wrap(1,
            biasLayer.addRef(),
            new NthPowerActivationLayer().setPower(2)).map(values)
    )
        .scaleInPlace(1.0 / values.size())
        .mapAndFree(v -> Math.pow(v, 0.5));
    logger.debug("Scale: " + scales.prettyPrint());

    List<Coordinate> coordList = scales.coordStream(true).sorted(Comparator.comparing(c -> -scales.get(c))).limit(50).collect(Collectors.toList());
    Map<String, Tensor> vectors = tensorMap.entrySet().stream().collect(Collectors.toMap(e -> e.getKey(), e -> {
      Tensor fullSignal = e.getValue();
      return new Tensor(coordList.size()).setByCoord(c2 -> fullSignal.get(coordList.get(c2.getIndex())));
    }));

    ProjectorUtil.browseProjector(ProjectorUtil.publishProjector(vectors));
  }

  @Test
  public void trainModel() throws IOException {
    List<String[]> rows = loadIndexFile();
    Map<Integer, Map<Integer, Tensor>> indexedData = toTensorMap(rows);
    PipelineNetwork classfier = classifierBase(indexedData);
    List<Tensor[]> indexData = rows.stream().map(e -> {
      return new Tensor[]{
          new Tensor(2).set(Integer.parseInt(e[0]), 1),
          new Tensor(SerialPrecision.Float.parse(e[2]), featureDims)
      };
    }).collect(Collectors.toList());
    classfier.wrap(pretrain(classfier, indexData));
    classfier.wrap(new SoftmaxLayer());
    Tensor[][] trainingData = IntStream.range(0, indexData.size()).mapToObj(i -> new Tensor[]{
        indexData.get(i)[1],
        indexData.get(i)[0]
    }).toArray(i -> new Tensor[i][]);
    double trainingResult = IterativeTrainer.wrap(new ArrayTrainable(trainingData, new SimpleLossNetwork(classfier, new EntropyLossLayer())))
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
    @NotNull Function<String, Future<Tensor>> languageTransform = getLanguageTransform(1);
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

  @NotNull
  protected PipelineNetwork classifierBase(Map<Integer, Map<Integer, Tensor>> indexedData) {
    Map<Integer, PipelineNetwork> networks = indexedData.entrySet().stream().collect(Collectors.toMap(e -> e.getKey(), entry -> {
      Map<Integer, Tensor> group = entry.getValue();
      List<Tensor> values = new ArrayList<>(group.values());
      Tensor avg = TestUtil.avg(values);
      BiasLayer biasLayer = new BiasLayer(avg.getDimensions()).set(avg.scaleInPlace(-1));
      Tensor scales = TestUtil.sum(PipelineNetwork.wrap(1, biasLayer.addRef(), new NthPowerActivationLayer().setPower(2)).map(values));
      Tensor scaleConst = scales
          .scaleInPlace(1.0 / values.size())
          .mapAndFree(v -> Math.pow(v, -0.5));
      PipelineNetwork net = new PipelineNetwork(1);
      net.wrap(new ProductInputsLayer(), net.wrap(biasLayer,
          net.wrap(new AssertDimensionsLayer(1, 24, 2, 16, 1, 64), net.getInput(0))
      ), net.constValue(scaleConst));
      net.wrap(new NthPowerActivationLayer().setPower(2));
      net.wrap(new SumReducerLayer());
      net.wrap(new NthPowerActivationLayer().setPower(0.5));
      return net;
    }));

    PipelineNetwork classfier = new PipelineNetwork(1);
    DAGNode input = classfier.wrap(new AssertDimensionsLayer(1, 24, 2, 16, 1, 64), classfier.getInput(0)); //classfier.getInput(0);
    classfier.wrap(new TensorConcatLayer(),
        classfier.add(networks.get(0), input),
        classfier.add(networks.get(1), input));
    return classfier;
  }

  protected Map<Integer, Map<Integer, Tensor>> toTensorMap(List<String[]> rows) {
    return rows.stream().collect(Collectors.groupingBy((String[] row) -> {
      return Integer.parseInt(row[0]);
    }, Collectors.toMap((String[] row) -> {
      return Integer.parseInt(row[1]);
    }, (String[] row) -> {
      return new Tensor(SerialPrecision.Float.parse(row[2]), featureDims);
    })));
  }

  protected List<String[]> loadIndexFile() throws IOException {
    File file = new File(base, "index.csv");
    return Arrays.stream(FileUtils.readFileToString(file, "UTF-8").split("\n"))
        .map(s -> s.split(",", 4)).collect(Collectors.toList());
  }

}


