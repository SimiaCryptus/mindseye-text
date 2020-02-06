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
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.*;
import com.simiacryptus.text.gpt2.GPT2Codec;
import com.simiacryptus.text.gpt2.GPT2Model;
import com.simiacryptus.text.gpt2.GPT2Util;
import com.simiacryptus.util.JsonUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.function.Function;

public class ClassifyUtil {

  protected static final Logger logger = LoggerFactory.getLogger(ClassifyUtil.class);

  @Nonnull
  public static Function<String, Future<Tensor>> getLanguageTransform(int threads) {
    ExecutorService pool = Executors.newFixedThreadPool(threads);
    ThreadLocal<GPT2Model> gpt2ModelThreadLocal = new ThreadLocal<GPT2Model>() {
      @Nonnull
      @Override
      protected GPT2Model initialValue() {
        return GPT2Util.getModel_345M();
      }
    };
    ThreadLocal<GPT2Codec> gpt2CodecThreadLocal = new ThreadLocal<GPT2Codec>() {
      @Nonnull
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
        return RefIntStream.range(0, prediction.length).mapToDouble(i -> prediction[i]).map(x -> x * Math.log(x)).sum()
            / Math.log(2);
      }).toArray();
      Tensor state = TFIO.getTensor(copy.state());
      int slice = state.getDimensions()[4] - 1;
      Tensor tensor = new Tensor(1, 24, 2, 16, 1, 64);
      tensor.setByCoord(c -> {
        int[] coords = c.getCoords();
        coords[4] = slice;
        return state.get(coords);
      });
      return tensor.addRef();
    });
  }

  public static <K, V, U> RefMap<K, U> mapValues(@Nonnull RefMap<K, V> indexedData, @Nonnull Function<V, U> fn) {
    return indexedData.entrySet().stream().collect(RefCollectors.toMap(e -> e.getKey(), e -> fn.apply(e.getValue())));
  }

  @Nonnull
  public static Tuple2<PipelineNetwork, SerializableFunction<Tensor, Tensor>> buildClassifier(
      @Nonnull RefMap<? extends Integer, ? extends RefCollection<? extends Tensor>> collect) {
    RefMap<? extends Integer, ? extends TensorStats> statsMap = mapValues(collect, entry -> TensorStats.create(entry));
    statsMap.values().forEach(tensorStats -> {
      logger.debug("Averages Histogram: " + JsonUtil.toJson(getHistogramList(tensorStats.avg.getData(), 10, 10)));
      assert tensorStats.scale != null;
      logger.debug("Scales Histogram: " + JsonUtil.toJson(getHistogramList(tensorStats.scale.getData(), 10, 10)));
    });
    int primaryKey = 0;
    int secondaryKey = 1;
    int maxDims = 1000;
    int[] dimensions = statsMap.get(primaryKey).scale.getDimensions();
    if (Tensor.length(dimensions) > maxDims) {
      SerializableFunction<Tensor, Tensor> tensorFunction = Tensor
          .select(mostSignifigantCoords(statsMap.get(primaryKey), statsMap.get(secondaryKey), maxDims));
      Tuple2<PipelineNetwork, SerializableFunction<Tensor, Tensor>> subClassifier = buildClassifier(mapValues(collect,
          values -> new RefArrayList<>(values.stream().map(tensorFunction).collect(RefCollectors.toList()))));
      return new Tuple2<>(subClassifier._1, x -> subClassifier._2.apply(tensorFunction.apply(x)));
    } else {
      return new Tuple2<>(buildClassifierFromStats(statsMap), x -> x);
    }
  }

  @Nonnull
  public static PipelineNetwork buildClassifierFromStats(@Nonnull RefMap<? extends Integer, ? extends TensorStats> statsMap) {
    int[] dimensions = RefUtil.get(statsMap.values().stream().findAny()).avg.getDimensions();
    RefMap<Integer, PipelineNetwork> networks = statsMap.entrySet().stream()
        .collect(RefCollectors.toMap(e -> e.getKey(), entry -> {
          TensorStats tensorStats = entry.getValue();
          PipelineNetwork net = new PipelineNetwork(1);
          net.add(new ProductInputsLayer(),
              net.add(tensorStats.biasLayer, net.add(new AssertDimensionsLayer(dimensions), net.getInput(0))),
              net.constValue(tensorStats.scale));
          NthPowerActivationLayer nthPowerActivationLayer1 = new NthPowerActivationLayer();
          nthPowerActivationLayer1.setPower(2);
          net.add(nthPowerActivationLayer1.addRef());
          net.add(new SumReducerLayer());
          NthPowerActivationLayer nthPowerActivationLayer = new NthPowerActivationLayer();
          nthPowerActivationLayer.setPower(0.5);
          net.add(nthPowerActivationLayer.addRef());
          return net;
        }));

    PipelineNetwork classfier = new PipelineNetwork(1);
    DAGNode input = classfier.add(new AssertDimensionsLayer(dimensions), classfier.getInput(0)); //classfier.getInput(0);
    classfier.add(new TensorConcatLayer(), classfier.add(networks.get(0), input),
        classfier.add(networks.get(1), input));
    return classfier;
  }

  @Nonnull
  public static Coordinate[] mostSignifigantCoords(@Nonnull TensorStats a, @Nonnull TensorStats b, int n) {
    return a.avg.coordStream(true)
        .sorted(RefComparator.comparingDouble(c -> {
          double avgA = a.avg.get(c);
          double avgB = b.avg.get(c);
          assert a.scale != null;
          double stddevA = a.scale.get(c);
          assert b.scale != null;
          double stddevB = b.scale.get(c);
          return -Math.log(stddevA / stddevB)
              + (Math.pow(stddevB, 2) + Math.pow(avgB - avgA, 2)) / (2 * Math.pow(stddevA, 2)) - 0.5;
        })).limit(n).toArray(i -> new Coordinate[i]);
  }

  public static RefList<String> getHistogramList(@Nonnull double[] data, int granularity, int base) {
    return getHistogram(data, granularity, base).entrySet().stream()
        .sorted(RefComparator.comparingDouble(x -> x.getKey()))
        .map(x -> RefString.format("%s=%s", x.getKey(), x.getValue()))
        .collect(RefCollectors.toList());
  }

  protected static RefMap<Double, Long> getHistogram(@Nonnull double[] data, int granularity, int base) {
    return RefArrays.stream(data).mapToObj(x -> {
      return Math.exp(Math.log(base) * Math.round(granularity * Math.log(x) / Math.log(base)) / granularity);
    }).collect(RefCollectors.groupingBy(x -> x, RefCollectors.counting()));
  }
}
