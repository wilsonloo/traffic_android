package com.wilson_loo.traffic_label.tflite;

import android.app.Activity;
import android.graphics.Bitmap;
import android.graphics.RectF;

import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.image.ops.Rot90Op;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ClassifierYoloV3 extends Classifier {
    /** Float MobileNet requires additional normalization of the used input. */
    private static final float IMAGE_MEAN = 0.0f;

    private static final float IMAGE_STD = 255.0f;

    /**
     * Float model does not need dequantization in the post-processing. Setting mean and std as 0.0f
     * and 1.0f, repectively, to bypass the normalization.
     */
    private static final float PROBABILITY_MEAN = 0.0f;

    private static final float PROBABILITY_STD = 1.0f;

    // Only return this many results.
    private static final int NUM_DETECTIONS = 10;

    // 置信度阀值
    private float mScoreThreshold = 0.8f;
    public void setScoreThreshold(float threshold){mScoreThreshold = threshold;}

    private static final int LARGE_SCALE = 76;
    private static final int MIDDLE_SCALE = 38;
    private static final int SMALL_SCALE = 19;

    private int mBoxId = 0;

    // outputLocations: array of shape [Batchsize, NUM_DETECTIONS,4]
    // contains the location of detected boxes
    private float[][][][][] mPred_lbbox;
    private float[][][][][] mPred_mbbox;
    private float[][][][][] mPred_sbbox;

    // outputClasses: array of shape [Batchsize, NUM_DETECTIONS]
    // contains the classes of detected boxes
    private float[][] outputClasses;

    // outputScores: array of shape [Batchsize, NUM_DETECTIONS]
    // contains the scores of detected boxes
    private float[][] outputScores;

    // numDetections: array of shape [Batchsize]
    // contains the number of detected boxes
    private float[] numDetections;


    protected ClassifierYoloV3(Activity activity, Device device, int numThreads) throws IOException {
        super(activity, device, numThreads);
    }

    @Override
    public String getModelPath() {
        return "yolov3_608_66_test_loss_2.5108.tflite";
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

    public List<Recognition> RecognizeImage(int faceId, final Bitmap bitmap, int sensorOrientation) {
        // 将原始图片载入成tensorflow 的图片张量
        mInputTensorImage.load(bitmap);
        int originalWidth = bitmap.getWidth();
        int originalHeight = bitmap.getHeight();
        float ratio = Math.min(imageSizeX * 1.0f / originalWidth, imageSizeY * 1.0f / originalHeight);
        int numRotation = sensorOrientation / 90;
        // TODO(b/143564309): Fuse ops inside ImageProcessor.
        ImageProcessor imageProcessor = new ImageProcessor.Builder()
                .add(new ResizeWithCropOrPadOp(imageSizeX, imageSizeX))
                .add(new ResizeOp(imageSizeX, imageSizeY, ResizeOp.ResizeMethod.BILINEAR))
                .add(new Rot90Op(numRotation))
                .add(getPreprocessNormalizeOp())
                .build();
        TensorImage inputImageBuffer = imageProcessor.process(mInputTensorImage);

        // 进行实际的预测
        // tflite.run(inputImageBuffer.getBuffer(), mOutputProbabilityBuffer.getBuffer().rewind());
        // 转换为 各个标签对应的概率
        // TensorLabel labelHelper = new TensorLabel(labels, mProbabilityProcessor.process(mOutputProbabilityBuffer));
        // Map<String/*标签*/, Float /*预测成该标签的概率*/> labeledProbability = labelHelper.getMapWithFloatValue();

        mPred_lbbox = new float[1][LARGE_SCALE][LARGE_SCALE][3][5+66];
        mPred_mbbox = new float[1][MIDDLE_SCALE][MIDDLE_SCALE][3][5+66];
        mPred_sbbox = new float[1][SMALL_SCALE][SMALL_SCALE][3][5+66];
        outputClasses = new float[1][NUM_DETECTIONS];
        outputScores = new float[1][NUM_DETECTIONS];
        numDetections = new float[1];
        Object[] inputArray = {inputImageBuffer.getBuffer()};
        Map<Integer, Object> outputMap = new HashMap<>();
        outputMap.put(0, mPred_lbbox);
        outputMap.put(1, mPred_mbbox);
        outputMap.put(2, mPred_sbbox);
//        outputMap.put(3, outputClasses);
//        outputMap.put(4, outputScores);
//        outputMap.put(5, numDetections);
        tflite.runForMultipleInputsOutputs(inputArray, outputMap);

        mBoxId = 0;
        final ArrayList<Recognition> recognitions = new ArrayList<>(LARGE_SCALE);
        recognitions.clear();
        postprocessBoxes(mPred_lbbox, LARGE_SCALE, ratio, recognitions);
        postprocessBoxes(mPred_mbbox, MIDDLE_SCALE, ratio, recognitions);
        postprocessBoxes(mPred_sbbox, SMALL_SCALE, ratio, recognitions);

        return recognitions;
    }

    private void postprocessBoxes(final float[][][][][] bbox, int scale, float ratio, ArrayList<Recognition> recognitions) {
        for (int i = 0; i < scale; ++i){
            for (int j = 0; j < scale; ++j) {
                for (int k = 0; k < 3; ++k) {
                    // 只处理置信度满足的框框
                    float score = bbox[0][i][j][k][4];
                    if(score >= mScoreThreshold) {
                        float x = bbox[0][i][j][k][0] / ratio;
                        float y = bbox[0][i][j][k][1] / ratio;
                        float width = bbox[0][i][j][k][2] / ratio;
                        float height = bbox[0][i][j][k][3] / ratio;

                        int label = getTopOneLabel(bbox[0][i][j][k]);
                        final RectF detection = new RectF(x-width/2, y-height/2, x + width/2, y + height/2);
                        recognitions.add(new Recognition("" + mBoxId, "label-" + label, score, detection));
                        ++mBoxId;
                    }
                }
            }
        }
    }

    private int getTopOneLabel(final float[] probilities){
        int index = -1;
        float maxProbility = -1;
        for (int k = 5; k < probilities.length; ++k){
            if(probilities[k] > maxProbility){
                maxProbility = probilities[k];
                index = k;
            }
        }

        if (index == -1){
            return -1;
        }

        return index - 5;
    }
}
