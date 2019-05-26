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
import com.simiacryptus.util.Util;
import org.apache.commons.io.FileUtils;
import org.jetbrains.annotations.NotNull;
import org.junit.Test;

import javax.annotation.Nonnull;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.security.KeyManagementException;
import java.security.NoSuchAlgorithmException;
import java.util.Arrays;
import java.util.TreeSet;
import java.util.stream.Collectors;

public class TextGenerationDemo extends NotebookReportBase {
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
  public void generateShakespeare() throws URISyntaxException, NoSuchAlgorithmException, IOException, KeyManagementException {
    TextGenerator textGenerator = getTextGenerator(
        "a-zA-Z01-9\\. ,\\n\\-!\\?",
        "http://www.gutenberg.org/cache/epub/1785/pg1785.txt");
    run(log->{
      for (int i = 0; i < 100; i++) {
        log.eval(()->{
          return textGenerator.generateText(s -> s.split("\\s+").length<100 || !s.endsWith("."), "ACT II. SCENE I.\n");
        });
      }
    });
  }

  @Test
  public void generateChildrensText() throws URISyntaxException, NoSuchAlgorithmException, IOException, KeyManagementException {
    TextGenerator textGenerator = getTextGenerator(
        "a-zA-Z \\n\\.\\!\\?",
        "http://www.mit.edu/~ecprice/wordlist.10000");
    run(log->{
      for (int i = 0; i < 10; i++) {
        log.eval(()->{
          return textGenerator.generateText(s -> s.split("\\s+").length<100 || !s.endsWith("."), "The Cat in the Hat\n");
        });
      }
    });
  }

  @NotNull
  protected static TextGenerator getTextGenerator(String characterWhitelist, String wordlistUrl) throws IOException, NoSuchAlgorithmException, KeyManagementException, URISyntaxException {
    TreeSet<String> wordList = new TreeSet<>(
        Arrays.stream(FileUtils.readFileToString(Util.cacheFile(new URI(wordlistUrl)), "UTF-8").split("\\s+"))
            .map(x->x.trim().toLowerCase()).collect(Collectors.toSet())
    );
    TextGenerator textGenerator = GPT2Util.get345M().setVerbose(false);
    textGenerator.getModel().setFilterFn((prefix,txt)->{
      if(txt.matches(".*[^" + characterWhitelist + "].*")) return false;
      String[] words = txt.split("[^\\w]+");
      for (int i = 0; i < words.length; i++) {
        String word = words[i].toLowerCase();
        if(null==word) continue;
        if(word.isEmpty()) continue;
        boolean isWordAllowed;
        if(i < words.length-1) isWordAllowed = wordList.contains(word);
        else {
          String floor = wordList.floor(word);
          String ceiling = wordList.ceiling(word);
          isWordAllowed = wordList.contains(word) || (null != floor && floor.startsWith(word)) || (null != ceiling && ceiling.startsWith(word));
        }
        if (!isWordAllowed) {
          return false;
        }
      }
      return true;
    });
    return textGenerator;
  }


}


