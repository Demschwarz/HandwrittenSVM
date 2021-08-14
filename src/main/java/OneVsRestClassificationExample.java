/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//package org.apache.ignite.examples.ml.multiclass;

import java.io.IOException;
import java.util.Arrays;
import javax.cache.Cache;

import org.apache.commons.math3.util.Precision;
import org.apache.ignite.Ignite;
import org.apache.ignite.IgniteCache;
import org.apache.ignite.Ignition;
import org.apache.ignite.cache.query.QueryCursor;
import org.apache.ignite.cache.query.ScanQuery;
import dataFiles.MLSandboxDatasets;
import dataFiles.SandboxMLCache;
import org.apache.ignite.ml.dataset.feature.extractor.impl.DummyVectorizer;
import org.apache.ignite.ml.math.primitives.vector.Vector;
import org.apache.ignite.ml.multiclass.MultiClassModel;
import org.apache.ignite.ml.multiclass.OneVsRestTrainer;
import org.apache.ignite.ml.preprocessing.Preprocessor;
import org.apache.ignite.ml.preprocessing.minmaxscaling.MinMaxScalerTrainer;
import org.apache.ignite.ml.svm.SVMLinearClassificationModel;
import org.apache.ignite.ml.svm.SVMLinearClassificationTrainer;

/**
 * Run One-vs-Rest multi-class classification trainer ({@link OneVsRestTrainer}) parametrized by binary SVM classifier
 * ({@link SVMLinearClassificationTrainer}) over distributed dataset to build two models: one with min-max scaling and
 * one without min-max scaling.
 * <p>
 * Code in this example launches Ignite grid and fills the cache with test data points (preprocessed
 * <a href="https://archive.ics.uci.edu/ml/datasets/Glass+Identification">Glass dataset</a>).</p>
 * <p>
 * After that it trains two One-vs-Rest multi-class models based on the specified data - one model is with min-max
 * scaling and one without min-max scaling.</p>
 * <p>
 * Finally, this example loops over the test set of data points, applies the trained models to predict what cluster does
 * this point belong to, compares prediction to expected outcome (ground truth), and builds
 * <a href="https://en.wikipedia.org/wiki/Confusion_matrix">confusion matrix</a>.</p>
 * <p>
 * You can change the test data used in this example and re-run it to explore this algorithm further.</p> NOTE: the
 * smallest 3rd class could not be classified via linear SVM here.
 */
public class OneVsRestClassificationExample {
    /**
     * Run example.
     */
    public static void main(String[] args) throws IOException {
        System.out.println();
        System.out.println(">>> One-vs-Rest SVM Multi-class classification model over cached dataset usage example started.");
        // Start ignite grid.
        try (Ignite ignite = Ignition.start("config/default-config.xml")) {
            System.out.println(">>> Ignite grid started.");

            IgniteCache<Integer, Vector> dataCache = null;
            try {
                long time;
                time = System.currentTimeMillis();
                System.out.print("Start fillCacheWith .. ");
//                dataCache = new SandboxMLCache(ignite).fillCacheWith(MLSandboxDatasets.MNIST_TRAIN_5);
//                dataCache = new SandboxMLCache(ignite).fillCacheWith(MLSandboxDatasets.MNIST_TRAIN_8);
//                dataCache = new SandboxMLCache(ignite).fillCacheWith(MLSandboxDatasets.MNIST_TRAIN_10);
//                dataCache = new SandboxMLCache(ignite).fillCacheWith(MLSandboxDatasets.MNIST_TRAIN_12);
                dataCache = new SandboxMLCache(ignite).fillCacheWith(MLSandboxDatasets.MNIST_TRAIN_15);
                System.out.println("complete, dataCache.size() = " + dataCache.size() + ", time = " + (System.currentTimeMillis() - time) / 1000.0);

                time = System.currentTimeMillis();
                OneVsRestTrainer<SVMLinearClassificationModel> trainer
                        = new OneVsRestTrainer<>(new SVMLinearClassificationTrainer()
                        .withAmountOfIterations(20)
                        .withAmountOfLocIterations(50)
                        .withLambda(0.2)
                        .withSeed(1234L)
                );

                System.out.print("Start mdl trainer.fit .. ");
                MultiClassModel<SVMLinearClassificationModel> mdl = trainer.fit(
                        ignite,
                        dataCache,
                        new DummyVectorizer<Integer>().labeled(0)
                );
                System.out.println("complete, time = " + (System.currentTimeMillis() - time) / 1000.0);
            } finally {
                if (dataCache != null)
                    dataCache.destroy();
            }
        } finally {
            System.out.flush();
        }
    }
}
