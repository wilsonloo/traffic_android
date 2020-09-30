package com.arcsoft.trafficLabel.tflite;

import android.app.Activity;
import android.graphics.Bitmap;

import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.image.ops.Rot90Op;
import org.tensorflow.lite.support.label.TensorLabel;

import java.io.IOException;
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

    protected ClassifierYoloV3(Activity activity, Device device, int numThreads) throws IOException {
        super(activity, device, numThreads);
    }

    @Override
    public String getModelPath() {
        return "yolov3_608_66_test_loss_2.5108.tflite";
    }

    @Override
    public String getLabelPath() {
        return null;
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
        int cropSize = Math.min(bitmap.getWidth(), bitmap.getHeight());
        int numRotation = sensorOrientation / 90;
        // TODO(b/143564309): Fuse ops inside ImageProcessor.
        ImageProcessor imageProcessor = new ImageProcessor.Builder()
                .add(new ResizeWithCropOrPadOp(cropSize, cropSize))
                .add(new ResizeOp(imageSizeX, imageSizeY, ResizeOp.ResizeMethod.BILINEAR))
                .add(new Rot90Op(numRotation))
                .add(getPreprocessNormalizeOp())
                .build();
        TensorImage inputImageBuffer = imageProcessor.process(mInputTensorImage);
        // 进行实际的预测
        tflite.run(inputImageBuffer.getBuffer(), mOutputProbabilityBuffer.getBuffer().rewind());
        // 转换为 各个标签对应的概率
        Map<String/*标签*/, Float /*预测成该标签的概率*/> labeledProbability =
                new TensorLabel(labels, mProbabilityProcessor.process(mOutputProbabilityBuffer))
                        .getMapWithFloatValue();
        // 获取前几项结果
        return getTopKProbability(labeledProbability, 1);
    }

}
