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

import com.simiacryptus.mindseye.test.NotebookReportBase;
import org.junit.Test;
import org.tensorflow.framework.GraphDef;

import javax.annotation.Nonnull;
import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;
import java.util.UUID;

public class UserTests extends NotebookReportBase {
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
  public void generateUnconditionalText() {
    run(log->{
      TextGenerator textGenerator = GPT2Util.get345M().setVerbose(false);
      for (int i = 0; i < 10; i++) {
        log.eval(()->{
          return textGenerator.generateText(500);
        });
      }
    });
  }

  @Test
  public void generateConditionalText() {
    run(log->{
      TextGenerator textGenerator = GPT2Util.get345M().setVerbose(false);
      for (int i = 0; i < 10; i++) {
        log.eval(()->{
          return textGenerator.generateText(500, "Hello");
        });
      }
    });
  }




}


