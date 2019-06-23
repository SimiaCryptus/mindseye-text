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
import com.simiacryptus.text.MinEntropyWrapper;
import com.simiacryptus.text.TemperatureWrapper;
import com.simiacryptus.text.TextGenerator;
import org.apache.commons.io.IOUtils;
import org.jsoup.Jsoup;
import org.junit.Test;

import javax.annotation.Nonnull;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.security.KeyManagementException;
import java.security.NoSuchAlgorithmException;
import java.util.Arrays;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.IntStream;

public class TextGenerationDemo extends NotebookReportBase {
  public static final String[] seeds_quotes = {
      "\"The greatest glory in living lies not in never falling, but in rising every time we fall.\" -Nelson Mandela",
      "\"If life were predictable it would cease to be life, and be without flavor.\" -Eleanor Roosevelt",
      "\"Life is what happens when you're busy making other plans.\" -John Lennon"
  };
  public static final String[] seeds_headlines = {
      "Trump tariffs threaten to drown gains from tax cuts",
      "SpaceX puts up 60 internet satellites",
      "Millions expected to hit the road for Memorial Day Weekend",
      "Food expiration labels are confusing and the FDA wants to change that",
      "In our opinion: This is the making of a retirement crisis"
  };
  public static final String[] seeds_fables = {
      "Wolf, meeting with a Lamb astray from the fold, resolved not to lay violent hands on him, but to find some plea to justify to the Lamb the Wolf's right to eat him. He thus addressed him: \"Sirrah, last year you grossly insulted me.\" \"Indeed,\" bleated the Lamb in a mournful tone of voice, \"I was not then born.\" Then said the Wolf, \"You feed in my pasture.\" \"No, good sir,\" replied the Lamb, \"I have not yet tasted grass.\" Again said the Wolf, \"You drink of my well.\" \"No,\" exclaimed the Lamb, \"I never yet drank water, for as yet my mother's milk is both food and drink to me.\" Upon which the Wolf seized him and ate him up, saying, \"Well! I won't remain supperless, even though you refute every one of my imputations.\" The tyrant will always find a pretext for his tyranny.",
      "A Bat who fell upon the ground and was caught by a Weasel pleaded to be spared his life. The Weasel refused, saying that he was by nature the enemy of all birds. The Bat assured him that he was not a bird, but a mouse, and thus was set free. Shortly afterwards the Bat again fell to the ground and was caught by another Weasel, whom he likewise entreated not to eat him. The Weasel said that he had a special hostility to mice. The Bat assured him that he was not a mouse, but a bat, and thus a second time escaped.\nIt is wise to turn circumstances to good account.",
      "An Ass having heard some Grasshoppers chirping, was highly enchanted; and, desiring to possess the same charms of melody, demanded what sort of food they lived on to give them such beautiful voices. They replied, \"The dew.\" The Ass resolved that he would live only upon dew, and in a short time died of hunger.",
      "A Kion was awakened from sleep by a Mouse running over his face. Rising up angrily, he caught him and was about to kill him, when the Mouse piteously entreated, saying: \"If you would only spare my life, I would be sure to repay your kindness.\" The Lion laughed and let him go. It happened shortly after this that the Lion was caught by some hunters, who bound him by st ropes to the ground. The Mouse, recognizing his roar, came gnawed the rope with his teeth, and set him free, exclaim\n" +
          "\n" +
          "\"You ridiculed the idea of my ever being able to help you, expecting to receive from me any repayment of your favor; I now you know that it is possible for even a Mouse to con benefits on a Lion.\"",
      "A Charcoal-Burner carried on his trade in his own house. One day he met a friend, a Fuller, and entreated him to come and live with him, saying that they should be far better neighbors and that their housekeeping expenses would be lessened. The Fuller replied, \"The arrangement is impossible as far as I am concerned, for whatever I should whiten, you would immediately blacken again with your charcoal.\"\n" +
          "\n" +
          "Like will draw like. "
  };

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
    run(log -> {
      TextGenerator textGenerator = GPT2Util.get345M().setVerbose(false);
      for (int i = 0; i < 10; i++) {
        log.eval(() -> {
          return textGenerator.generateText(500);
        });
      }
    });
  }

  @Test
  public void writeFables() {
    run(log -> {
      int articles = 100;
      TextGenerator textGenerator = null;
      try {
        textGenerator = GPT2Util.getTextGenerator(
            GPT2Util.getTextGenerator(
                "\\w\\s\\.\\;\\,\\'\\\"\\-\\(\\)\\d\\n",
                new URI("http://www.mit.edu/~ecprice/wordlist.10000")
            ),
            Arrays.copyOf(seeds_fables, 2)
        );
      } catch (Exception e) {
        throw new RuntimeException(e);
      }
      textGenerator.setModel(new TemperatureWrapper(1.5, textGenerator.getModel()));
      textGenerator.setModel(new MinEntropyWrapper(5e-1, textGenerator.getModel()));
      textGenerator.feed("\n");
      for (int i = 0; i < articles; i++) {
        TextGenerator copy = textGenerator.copy();
        try {
          String headline = copy.generate(s -> s.length() < 10 || !s.endsWith("\n")).replace('\n', ' ').trim();
          log.h1(headline);
          log.p(headline + " " + IntStream.range(0, 15)
              .mapToObj(j -> copy.generate(s -> s.length() < 32 || (s.length() < 500 && !s.endsWith(".") && !s.contains(". ") && !s.contains("\n"))))
              .map(x -> x.replace('\n', ' ').trim())
              .reduce((a, b) -> a + " " + b).orElse(""));
        } catch (Throwable e) {
          logger.warn("Err", e);
        }
      }
    });
  }

  @Test
  public void writeFakeNews() {
    run(log -> {
      int articles = 100;
      TextGenerator textGenerator = null;
      try {
        textGenerator = GPT2Util.getTextGenerator(
            "\\w\\s\\.\\;\\,\\'\\\"\\-\\(\\)\\d\\n",
            new URI("http://www.mit.edu/~ecprice/wordlist.10000")
        );
        textGenerator.getModel();
        textGenerator = GPT2Util.getTextGenerator(
            textGenerator,
            seeds_headlines
        );
      } catch (Exception e) {
        throw new RuntimeException(e);
      }
      textGenerator.feed("\n");
      for (int i = 0; i < articles; i++) {
        TextGenerator copy = textGenerator.copy();
        String headline = copy.generate(s -> s.length() < 10 || !s.endsWith("\n")).replace('\n', ' ').trim();
        log.h1(headline);
        log.p(headline + " " + IntStream.range(0, 15)
            .mapToObj(j -> copy.generate(s -> s.length() < 32 || (s.length() < 500 && !s.endsWith(".") && !s.contains(". ") && !s.contains("\n"))))
            .map(x -> x.replace('\n', ' ').trim())
            .reduce((a, b) -> a + " " + b).orElse(""));
      }
    });
  }


  @Test
  public void generateCommentary() throws NoSuchAlgorithmException, IOException, KeyManagementException {
    TextGenerator textGenerator = GPT2Util.getTextGenerator(
        "",
//        "a-zA-Z01-9 ,;:\\-\\.\\!\\?",
        null
//        new URI("http://www.mit.edu/~ecprice/wordlist.10000")
    );
//    String url = "https://en.wikipedia.org/wiki/Special:Random";
    String url = "https://en.wikinews.org/wiki/Special:Random";
    String text = Jsoup.connect(url).followRedirects(true).get().select("p").text();
//    String text = ("");
    String sentancePattern = "([^\\.]{8,}\\.)";
//    String sentancePattern = "([^\n]{6,}\n)";
    Pattern pattern = Pattern.compile(sentancePattern);
    run(log -> {
      log.h2("Raw Text");
      log.p(text);
      log.h2("Sentence Analysis");
      Matcher matcher = pattern.matcher(text);
      try {
        while (matcher.find()) {
          String line = matcher.group(1).trim();
          log.h3(line);
          double totalEntropy = textGenerator.feed(line);
          log.p(String.format("%.3f bits, %.3f bits per word, %.3f bits per character", totalEntropy, totalEntropy / line.split("\\s+").length, totalEntropy / line.length()));
          for (int i = 0; i < 5; i++) {
            TextGenerator copy = textGenerator.copy();
            String generate = copy.generate(s -> !s.endsWith("."));
            copy.getModel().clear();
            log.p(generate);
          }
        }
      } catch (Exception e) {
        logger.warn("Error", e);
      }
    });
  }

  @Test
  public void generateCodeComments() throws NoSuchAlgorithmException, IOException, KeyManagementException, URISyntaxException {
    String url = "https://raw.githubusercontent.com/SimiaCryptus/tf-gpt-2/master/src/main/java/com/simiacryptus/text/gpt2/GPT2Model.java";
    TextGenerator textGenerator = GPT2Util.getTextGenerator(
        "a-zA-Z01-9 \\{\\}\\[\\]\\(\\)\\'\\\"\\n\\.\\!\\?",
        new URI("http://www.mit.edu/~ecprice/wordlist.10000")
    );
    run(log -> {
      try {
        Arrays.stream(IOUtils.toString(new URI(url), "UTF-8").split("\n"))
            .filter(x -> !x.trim().startsWith("/") && !x.trim().startsWith("*") && !x.trim().startsWith("import") && !x.trim().isEmpty()).forEach(line -> {
          log.h3(line);
          double totalEntropy = textGenerator.feed(line);
          log.p(String.format("%.3f bits, %.3f bits per word, %.3f bits per character", totalEntropy, totalEntropy / line.split("\\s+").length, totalEntropy / line.length()));
          for (int i = 0; i < 5; i++) {
            TextGenerator copy = textGenerator.copy();
            copy.feed("        // ");
            String generate = copy.generate(s -> s.length() < 8 || (!s.endsWith("\n") && s.length() < 128));
            copy.getModel().clear();
            log.p(generate);
          }
          textGenerator.feed("\n");
        });
      } catch (Exception e) {
        logger.warn("Error", e);
      }
    });
  }
}


