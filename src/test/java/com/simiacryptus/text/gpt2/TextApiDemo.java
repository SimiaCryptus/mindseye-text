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

import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.notebook.StringQuery;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefStream;
import com.simiacryptus.text.MinEntropyWrapper;
import com.simiacryptus.text.TemperatureWrapper;
import com.simiacryptus.text.TextGenerator;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.test.NotebookTestBase;
import org.jetbrains.annotations.NotNull;
import org.junit.jupiter.api.Test;

import javax.annotation.Nonnull;
import java.net.URI;
import java.util.function.Predicate;

public class TextApiDemo extends NotebookTestBase {

  @Nonnull
  @Override
  public ReportType getReportType() {
    return ReportType.Applications;
  }

  @Nonnull
  @Override
  protected Class<?> getTargetClass() {
    return GPT2Model.class;
  }

  @Test
  public void startAPI() {
    final NotebookOutput log = getLog();
    while (true) {
      StringQuery.SimpleStringQuery simpleStringQuery = new StringQuery.SimpleStringQuery(log);
      simpleStringQuery.setValue("\n");
      simpleStringQuery.setSubmitLabel("Go");
      simpleStringQuery.print();
      String priorInput = simpleStringQuery.get();
      String totalMessage = generate(priorInput);
      log.p("```" + totalMessage + "```");
    }
  }

  @NotNull
  public String generate(String priorInput) {
    final TextGenerator textGenerator = getTextGenerator("");
    textGenerator.feed(priorInput);
    Predicate<String> lineGenFn = s -> s.length() < 32 || s.length() < 500 && !s.endsWith(".") && !s.contains(". ") && !s.contains("\n");
    return toString(
        RefIntStream.range(0, 15).mapToObj(j -> textGenerator.generate(lineGenFn)).map(x -> toDisplayText(x))
    );
  }

  public @NotNull String toString(RefStream<@NotNull String> continued) {
    return continued.reduce((a, b) -> a + " " + b).orElse("");
  }

  @NotNull
  public String toDisplayText(String generate) {
    return generate.replace('\n', ' ').trim();
  }

  @NotNull
  public TextGenerator getTextGenerator(String... seeds) {
    final TextGenerator textGenerator;
    try {
      TextGenerator generator = GPT2Util.getTextGenerator("\\w\\s\\.\\;\\,\\'\\\"\\-\\(\\)\\d\\n",
          new URI("http://www.mit.edu/~ecprice/wordlist.10000"));
      textGenerator = GPT2Util.getTextGenerator(generator, seeds);
    } catch (Exception e) {
      throw Util.throwException(e);
    }
    textGenerator.setModel(new TemperatureWrapper(1.5, textGenerator.getModel()));
    textGenerator.setModel(new MinEntropyWrapper(5e-1, textGenerator.getModel()));
    return textGenerator;
  }

}
