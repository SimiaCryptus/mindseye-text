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

import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.java.BiasLayer;
import com.simiacryptus.mindseye.layers.java.NthPowerActivationLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.ref.lang.ReferenceCountingBase;
import com.simiacryptus.ref.wrappers.RefCollection;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;

public class TensorStats extends ReferenceCountingBase {
  protected static final Logger logger = LoggerFactory.getLogger(TensorStats.class);
  public BiasLayer biasLayer;
  @Nullable
  public Tensor scale;
  public Tensor avg;

  @Nonnull
  public static TensorStats create(@Nonnull RefCollection<? extends Tensor> values) {
    TensorStats self = new TensorStats();
    self.avg = TestUtil.avg(values);
    self.biasLayer = new BiasLayer(self.avg.getDimensions()).set(self.avg.scaleInPlace(-1));
    Tensor scales = TestUtil
        .sum(PipelineNetwork.build(1, self.biasLayer.addRef(), new NthPowerActivationLayer().setPower(2)).map(values));
    self.scale = scales.scaleInPlace(1.0 / values.size()).map(v -> Math.pow(v, -0.5));
    return self;
  }

  @Nullable
  public static @SuppressWarnings("unused")
  TensorStats[] addRefs(@Nullable TensorStats[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(TensorStats::addRef).toArray((x) -> new TensorStats[x]);
  }

  @Nullable
  public static @SuppressWarnings("unused")
  TensorStats[][] addRefs(@Nullable TensorStats[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(TensorStats::addRefs).toArray((x) -> new TensorStats[x][]);
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  TensorStats addRef() {
    return (TensorStats) super.addRef();
  }
}
