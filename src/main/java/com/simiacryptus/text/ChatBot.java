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

import com.simiacryptus.text.gpt2.GPT2Util;
import com.simiacryptus.util.test.SysOutInterceptor;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.PrintStream;
import java.net.URI;
import java.net.URISyntaxException;
import java.security.KeyManagementException;
import java.security.NoSuchAlgorithmException;
import java.util.Arrays;
import java.util.Scanner;
import java.util.function.BiFunction;

public class ChatBot {
  protected static final Logger logger = LoggerFactory.getLogger(ChatBot.class);
  private final TextGenerator root;
  private final String characterWhitelist = "a-zA-Z01-9,.'\"\\!\\@\\$\\&\\*\\(\\)\\#\\-\\=\\+/";
  private TextGenerator textGenerator;
  private String wordlist = ""; //"http://www.mit.edu/~ecprice/wordlist.10000";
  private double temperature = 1.0;
  private double minEntropy = 1e-1;
  private String[] seeds = new String[]{""};
  private boolean verbose = false;
  private int choicesToLog = 5;
  private int maxLength = 64;
  private String prefix = "";
  private String suffix = "";

  public ChatBot() throws URISyntaxException, KeyManagementException, NoSuchAlgorithmException, IOException {
    this.root = GPT2Util.getTextGenerator();
    this.textGenerator = init();
  }

  public static void main(String[] args) throws Exception {
    PrintStream out = SysOutInterceptor.ORIGINAL_OUT;
    ChatBot chatBot = new ChatBot();
    Scanner scanner = new Scanner(System.in);

    while (true) {
      try {
        out.print("\nUser> ");
        String nextLine = scanner.nextLine().trim();
        out.println("Processing...");
        String dialog = chatBot.dialog(nextLine);
        out.println("Computer> " + dialog);
      } catch (Exception e) {
        logger.warn("Error generating text", e);
        chatBot.reset();
      }
    }
  }

  @NotNull
  protected TextGenerator init() throws IOException, NoSuchAlgorithmException, KeyManagementException, URISyntaxException {
    TextGenerator textGenerator = GPT2Util.getTextGenerator(root.copy(), characterWhitelist, (null == wordlist || wordlist.isEmpty()) ? null : new URI(wordlist));
    textGenerator = GPT2Util.getTextGenerator(textGenerator, Arrays.asList(
//            SimpleModel.build(GPT2Util.getCodec_345M(), IOUtils.toString(new URI("http://classics.mit.edu/Aesop/fab.mb.txt"), "UTF-8"))
    ), seeds);
    textGenerator.setVerbose(verbose);
    textGenerator.setChoicesToLog(choicesToLog);
    textGenerator.setModel(new TemperatureWrapper(1.0 / temperature, textGenerator.getModel()));
    textGenerator.setModel(new MinEntropyWrapper(minEntropy, textGenerator.getModel()));
    BiFunction<String, String, Boolean> filterFn = textGenerator.getModel().getFilterFn();
    textGenerator.getModel().setFilterFn((s, s2) -> {
      if (s.endsWith("\n") && s2.startsWith("\n")) return false;
      return filterFn.apply(s, s2);
    });
    return textGenerator;
  }

  public String dialog(String nextLine) throws URISyntaxException, KeyManagementException, NoSuchAlgorithmException, IOException {
    if (!nextLine.equals(nextLine.trim())) return dialog(nextLine.trim());
    if (nextLine.equals("reset")) {
      textGenerator = init();
      return "AI State Reset";
    } else if (nextLine.equals("help")) {
      return "temp=X - Set Temperature\n" +
          "ent=X - Set Minimum Choice Entropy\n" +
          "verbose=X - Set Verbosity\n" +
          "choice=X - Set Number of Alternates to log\n" +
          "length=X - Set Length\n" +
          "wordlist=X - Set wordlist url";
    } else {
      String[] split = nextLine.split("=");
      if (split.length == 2)
        if (split[0].toLowerCase().startsWith("temp")) {
          temperature = Double.parseDouble(split[1]);
          textGenerator = init();
          return String.format("Temp=%s; AI State Reset", temperature);
        } else if (split[0].toLowerCase().startsWith("ent")) {
          minEntropy = Double.parseDouble(split[1]);
          textGenerator = init();
          return String.format("minEntropy=%s; AI State Reset", minEntropy);
        } else if (split[0].toLowerCase().equals("verbose")) {
          verbose = Boolean.parseBoolean(split[1]);
          textGenerator = init();
          return String.format("verbose=%s; AI State Reset", verbose);
        } else if (split[0].toLowerCase().startsWith("choice")) {
          choicesToLog = Integer.parseInt(split[1]);
          textGenerator = init();
          return String.format("choicesToLog=%s; AI State Reset", choicesToLog);
        } else if (split[0].toLowerCase().startsWith("length")) {
          maxLength = Integer.parseInt(split[1]);
          textGenerator = init();
          return String.format("maxLength=%s; AI State Reset", maxLength);
        } else if (split[0].toLowerCase().startsWith("wordlist")) {
          wordlist = (split[1]);
          textGenerator = init();
          return String.format("maxLength=%s; AI State Reset", wordlist);
        }

    }
    if (!nextLine.isEmpty()) {
      textGenerator.feed(prefix + nextLine + suffix);
    }
    return textGenerator.generate(s ->
        s.split("[^\\w]+").length < 4 ||
            s.split("[^\\w]+").length < maxLength && !s.contains("\n")
    ).trim();
  }

  public void reset() throws URISyntaxException, KeyManagementException, NoSuchAlgorithmException, IOException {
    textGenerator = init();
  }
}
