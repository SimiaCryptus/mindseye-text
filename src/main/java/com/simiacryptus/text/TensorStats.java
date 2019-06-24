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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collection;

public class TensorStats {
  protected static final Logger logger = LoggerFactory.getLogger(TensorStats.class);
  public BiasLayer biasLayer;
  public Tensor scale;
  public Tensor avg;

  public static TensorStats create(Collection<? extends Tensor> values) {
    TensorStats self = new TensorStats();
    self.avg = TestUtil.avg(values);
    self.biasLayer = new BiasLayer(self.avg.getDimensions()).set(self.avg.scaleInPlace(-1));
    Tensor scales = TestUtil.sum(PipelineNetwork.wrap(1, self.biasLayer.addRef(), new NthPowerActivationLayer().setPower(2)).map(values));
    self.scale = scales
        .scaleInPlace(1.0 / values.size())
        .mapAndFree(v -> Math.pow(v, -0.5));
    return self;
  }
}
