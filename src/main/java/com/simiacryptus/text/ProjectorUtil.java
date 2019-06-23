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

import com.amazonaws.services.s3.AmazonS3;
import com.amazonaws.services.s3.AmazonS3ClientBuilder;
import com.amazonaws.services.s3.model.CannedAccessControlList;
import com.amazonaws.services.s3.model.PutObjectRequest;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.Tensor;
import org.apache.commons.io.FileUtils;

import java.awt.*;
import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;
import java.net.URLEncoder;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.stream.Collectors;

public class ProjectorUtil {

  public static void browseProjector(URL configUrl) throws IOException, URISyntaxException {
    //Desktop.getDesktop().browse(configUrl.toURI());
    Desktop.getDesktop().browse(new URI("https://projector.tensorflow.org/?config=" + URLEncoder.encode(configUrl.toString(), "UTF-8")));
  }

  public static URL publishProjector(Map<String, Tensor> vectors) throws IOException {
    AmazonS3 s3 = AmazonS3ClientBuilder.standard().build();
    URL configUrl = publishProjector(vectors, s3);
    s3.shutdown();
    return configUrl;
  }

  public static URL publishProjector(Map<String, Tensor> vectors, AmazonS3 s3) throws IOException {
    String id = UUID.randomUUID().toString();
    return publishProjector(vectors, s3, id);
  }

  public static URL publishProjector(Map<String, Tensor> vectors, AmazonS3 s3, String id) throws IOException {
    List<Map.Entry<String, Tensor>> entries = vectors.entrySet().stream().collect(Collectors.toList());
    int reducedDims = entries.get(0).getValue().getData().length;

    String tensorsSrc = entries.stream().map(entry -> {
      return Arrays.stream(entry.getValue().getData()).mapToObj(x -> {
        return Double.toString(x);
      }).reduce((a, b) -> a + "\t" + b).orElse("");
    }).reduce((a, b) -> a + "\n" + b).orElse("");
    File tensorsFile = File.createTempFile(id, "tensors.tsv");
    tensorsFile.deleteOnExit();
    FileUtils.write(tensorsFile, tensorsSrc, "UTF-8");
    s3.putObject(new PutObjectRequest("simiacryptus", "projector/" + id + "/tensors.tsv", tensorsFile)
        .withCannedAcl(CannedAccessControlList.PublicRead));
    String metadataSrc = entries.stream().map(entry -> {
      return entry.getKey();
    }).reduce((a, b) -> a + "\n" + b).orElse("");
    File metadataFile = File.createTempFile(id, "metadata.tsv");
    metadataFile.deleteOnExit();
    FileUtils.write(metadataFile, metadataSrc, "UTF-8");
    s3.putObject(new PutObjectRequest("simiacryptus", "projector/" + id + "/metadata.tsv", metadataFile)
        .withCannedAcl(CannedAccessControlList.PublicRead));
    JsonObject envelopeObj = new JsonObject();
    JsonArray envelopeArray = new JsonArray();
    envelopeObj.add("embeddings", envelopeArray);
    JsonObject config = new JsonObject();
    envelopeArray.add(config);
    config.addProperty("tensorName", "text");
    JsonArray tensorShape = new JsonArray();
    tensorShape.add(entries.size());
    tensorShape.add(reducedDims);
    config.add("tensorShape", tensorShape);
    config.addProperty("tensorPath", s3.getUrl("simiacryptus", "projector/" + id + "/tensors.tsv").toString());
    config.addProperty("metadataPath", s3.getUrl("simiacryptus", "projector/" + id + "/metadata.tsv").toString());
    File configFile = File.createTempFile(id, "projector.json");
    configFile.deleteOnExit();
    FileUtils.write(configFile, envelopeObj.toString(), "UTF-8");
    s3.putObject(new PutObjectRequest("simiacryptus", "projector/" + id + "/projector.json", configFile)
        .withCannedAcl(CannedAccessControlList.PublicRead));
    return s3.getUrl("simiacryptus", "projector/" + id + "/projector.json");
  }
}
