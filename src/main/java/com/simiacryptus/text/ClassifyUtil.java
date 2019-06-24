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

package com.simiacryptus.text;

import com.simiacryptus.lang.SerializableFunction;
import com.simiacryptus.lang.Tuple2;
import com.simiacryptus.mindseye.lang.Coordinate;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.tensorflow.TFIO;
import com.simiacryptus.mindseye.layers.java.*;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.text.gpt2.GPT2Codec;
import com.simiacryptus.text.gpt2.GPT2Model;
import com.simiacryptus.text.gpt2.GPT2Util;
import com.simiacryptus.util.JsonUtil;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class ClassifyUtil {

  protected static final Logger logger = LoggerFactory.getLogger(ClassifyUtil.class);

  @NotNull
  public static Function<String, Future<Tensor>> getLanguageTransform(int threads) {
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

  public static <K, V, U> Map<K, U> mapValues(Map<K, V> indexedData, Function<V, U> fn) {
    return indexedData.entrySet().stream().collect(Collectors.toMap(e -> e.getKey(), e -> fn.apply(e.getValue())));
  }

  @NotNull
  public static Tuple2<PipelineNetwork, SerializableFunction<Tensor, Tensor>> buildClassifier(Map<? extends Integer, ? extends Collection<? extends Tensor>> collect) {
    Map<? extends Integer, ? extends TensorStats> statsMap = mapValues(collect, entry -> TensorStats.create(entry));
    statsMap.values().forEach(tensorStats -> {
      logger.debug("Averages Histogram: " + JsonUtil.toJson(getHistogramList(tensorStats.avg.getData(), 10, 10)));
      logger.debug("Scales Histogram: " + JsonUtil.toJson(getHistogramList(tensorStats.scale.getData(), 10, 10)));
    });
    int primaryKey = 0;
    int secondaryKey = 1;
    int maxDims = 1000;
    int[] dimensions = statsMap.get(primaryKey).scale.getDimensions();
    if (Tensor.length(dimensions) > maxDims) {
      SerializableFunction<Tensor, Tensor> tensorFunction = Tensor.select(mostSignifigantCoords(statsMap.get(primaryKey), statsMap.get(secondaryKey), maxDims));
      Tuple2<PipelineNetwork, SerializableFunction<Tensor, Tensor>> subClassifier = buildClassifier(mapValues(collect, values ->
          new ArrayList<>(values.stream().map(tensorFunction).collect(Collectors.toList()))));
      return new Tuple2<>(subClassifier._1, x -> subClassifier._2.apply(tensorFunction.apply(x)));
    } else {
      return new Tuple2<>(buildClassifierFromStats(statsMap), x -> x);
    }
  }

  @NotNull
  public static PipelineNetwork buildClassifierFromStats(Map<? extends Integer, ? extends TensorStats> statsMap) {
    int[] dimensions = statsMap.values().stream().findAny().get().avg.getDimensions();
    Map<Integer, PipelineNetwork> networks = statsMap.entrySet().stream().collect(Collectors.toMap(e -> e.getKey(), entry -> {
      TensorStats tensorStats = entry.getValue();
      PipelineNetwork net = new PipelineNetwork(1);
      net.wrap(new ProductInputsLayer(), net.wrap(tensorStats.biasLayer,
          net.wrap(new AssertDimensionsLayer(dimensions), net.getInput(0))
      ), net.constValue(tensorStats.scale));
      net.wrap(new NthPowerActivationLayer().setPower(2));
      net.wrap(new SumReducerLayer());
      net.wrap(new NthPowerActivationLayer().setPower(0.5));
      return net;
    }));

    PipelineNetwork classfier = new PipelineNetwork(1);
    DAGNode input = classfier.wrap(new AssertDimensionsLayer(dimensions), classfier.getInput(0)); //classfier.getInput(0);
    classfier.wrap(new TensorConcatLayer(),
        classfier.add(networks.get(0), input),
        classfier.add(networks.get(1), input));
    return classfier;
  }

  public static Coordinate[] mostSignifigantCoords(TensorStats a, TensorStats b, int n) {
    return a.avg.coordStream(true).sorted(Comparator.comparing(c -> {
      double avgA = a.avg.get(c);
      double avgB = b.avg.get(c);
      double stddevA = a.scale.get(c);
      double stddevB = b.scale.get(c);
      return -Math.log(stddevA / stddevB) + ((Math.pow(stddevB, 2) + Math.pow(avgB - avgA, 2)) / (2 * Math.pow(stddevA, 2))) - 0.5;
    })).limit(n).toArray(i -> new Coordinate[i]);
  }

  public static List<String> getHistogramList(double[] data, int granularity, int base) {
    return getHistogram(data, granularity, base)
        .entrySet()
        .stream()
        .sorted(Comparator.comparing(x -> x.getKey()))
        .map(x -> String.format("%s=%s", x.getKey(), x.getValue()))
        .collect(Collectors.toList());
  }

  protected static Map<Double, Long> getHistogram(double[] data, int granularity, int base) {
    return Arrays.stream(data).mapToObj(x -> {
      return Math.exp(Math.log(base) * Math.round(granularity * Math.log(x) / Math.log(base)) / granularity);
    }).collect(Collectors.groupingBy(x -> x, Collectors.counting()));
  }
}
