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
import java.util.HashSet;
import java.util.Iterator;
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

    private static final int SCORE_INDEX = 4;
    private static final int PROB_INDEX = 5;

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
        final ArrayList<float[]> boxes = new ArrayList<>();
        boxes.clear();

        final HashMap<Integer, ArrayList<float[]>> classesInImage = new HashMap<>();
        classesInImage.clear();

        concateBoxes(mPred_lbbox, LARGE_SCALE, ratio, boxes, classesInImage);
        concateBoxes(mPred_mbbox, MIDDLE_SCALE, ratio, boxes, classesInImage);
        concateBoxes(mPred_sbbox, SMALL_SCALE, ratio, boxes, classesInImage);

        final ArrayList<Recognition> recognitions = nms(boxes, classesInImage);
        return recognitions;
    }

    private void concateBoxes(final float[][][][][] bbox, int scale, float ratio, ArrayList<float[]> boxes, HashMap<Integer, ArrayList<float[]>> classesInImage) {
        for (int i = 0; i < scale; ++i){
            for (int j = 0; j < scale; ++j) {
                for (int k = 0; k < 3; ++k) {
                    // 只处理置信度满足的框框
                    float[] box = bbox[0][i][j][k];
                    float score = box[SCORE_INDEX];
                    if(score >= mScoreThreshold) {
                        for(int p = PROB_INDEX; p < box.length; ++p) {
                            if(score * box[p] >= 0.3f){ //mScoreThreshold) {
                                float x = box[0] / ratio;
                                float y = box[1] / ratio;
                                float width = box[2] / ratio;
                                float height = box[3] / ratio;

                                // 该框框的置信度可行
                                int label = p - PROB_INDEX;
                                float[] b = {x-width/2, y-height/2, x+width/2, y+height/2, score, label};
                                boxes.add(b);

                                if(!classesInImage.containsKey(label)) {
                                    classesInImage.put(label, new ArrayList<>());
                                }

                                classesInImage.get(label).add(b);
                            }
                        }
                    }
                }
            }
        }
    }

    private static int floatArgMax(ArrayList<float[]> list, int whichSlot){
        float maxValue = -1.0f;
        int maxIndex = -1;
        for(int k = 0; k < list.size(); ++k){
            if (list.get(k)[whichSlot] > maxValue){
                maxValue = list.get(k)[whichSlot];
                maxIndex = k;
            }
        }

        return maxIndex;
    }

    /*
     * @param bboxes: (xmin, ymin, xmax, ymax, score, class/label)
     * */
    private static void bboxes_iou(float[] bestBox, ArrayList<float[]> otherBoxes, float iouThreshold){
        if (otherBoxes.isEmpty()){
            return;
        }

        float bestArea = (bestBox[2] - bestBox[0]) * (bestBox[3] - bestBox[1]);
        for(int k = otherBoxes.size() - 1; k >= 0; --k){
            float[] otherBox = otherBoxes.get(k);
            float otherArea = (otherBox[2] - otherBox[0]) * (otherBox[3] - otherBox[1]);

            float[] leftUP = {Math.max(bestBox[0], otherBox[0]), Math.max(bestBox[1], otherBox[1])};
            float[] rightDown = {Math.min(bestBox[2], otherBox[2]), Math.min(bestBox[3], otherBox[3])};

            float[] interSection = {Math.max(rightDown[0] - leftUP[0], 0.0f), Math.max(rightDown[1] - leftUP[1], 0.0f)};
            float interArea = interSection[0] * interSection[1];
            float unionArea = bestArea + otherArea - interArea;
            float iou = unionArea <= 0 ? 0.0f : interArea / unionArea;
            if(iou >= iouThreshold){
                // 这个格子和 bestBox 有交集，标记为已处理
                otherBoxes.remove(k);
            }
        }
    }

    /*
    * @param bboxes: (xmin, ymin, xmax, ymax, score, class/label)
    * */
    private ArrayList<Recognition> nms(final ArrayList<float[]> bboxes, HashMap<Integer, ArrayList<float[]>> classesInImage){
        final ArrayList<Recognition> bestBoxes = new ArrayList<>();
        bestBoxes.clear();

        Iterator iter = classesInImage.entrySet().iterator();
        while (iter.hasNext()) {
            Map.Entry entry = (Map.Entry) iter.next();
            int label = (int) entry.getKey();
            ArrayList<float[]> labelBoxes = (ArrayList<float[]>) entry.getValue();
            while(!labelBoxes.isEmpty()) {
                int maxIndex = floatArgMax(labelBoxes, 4);
                if(maxIndex >= 0) {
                    float[] bestBox = labelBoxes.get(maxIndex);
                    labelBoxes.remove(maxIndex);

                    bboxes_iou(bestBox, labelBoxes, 0.45f);

                    final RectF detection = new RectF(bestBox[0], bestBox[1], bestBox[2], bestBox[3]);
                    bestBoxes.add(new Recognition("" + mBoxId, "label-" + label, bestBox[4], detection));
                    ++mBoxId;
                }
            }
        }

        return bestBoxes;
    }
}
