package com.wilson_loo.traffic_label.tflite;

import android.app.Activity;

import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.ops.NormalizeOp;

import java.io.IOException;

public class ClassifierFloatMobileNet extends Classifier {
    /** Float MobileNet requires additional normalization of the used input. */
    private static final float IMAGE_MEAN = 0.0f;

    private static final float IMAGE_STD = 255.0f;

    /**
     * Float model does not need dequantization in the post-processing. Setting mean and std as 0.0f
     * and 1.0f, repectively, to bypass the normalization.
     */
    private static final float PROBABILITY_MEAN = 0.0f;

    private static final float PROBABILITY_STD = 1.0f;


    public ClassifierFloatMobileNet(Activity activity, Device device, int numThreads)
        throws IOException
    {
        super(activity, device, numThreads);
    }

    @Override
    public String getModelPath() {
        return "emotion.model.mobilenetV2_x75_1.00_20200913.tflite";
//        return "emotion.tflite";
    }

    @Override
    public String getLabelPath() {
        return "labels.txt";
    }

    @Override
    protected TensorOperator getPreprocessNormalizeOp() {
        // return (x - IMAGE_MEAN) / IMAGE_STD
        return new NormalizeOp(IMAGE_MEAN, IMAGE_STD);
    }

    @Override
    protected TensorOperator getPostprocessNormalizeOp() {
        // return (x - PROBABILITY_MEAN) / PROBABILITY_STD
        return new NormalizeOp(PROBABILITY_MEAN, PROBABILITY_STD);
    }
}
