package com.wilson_loo.traffic_label.tflite;

import android.app.Activity;
import android.graphics.Bitmap;
import android.graphics.Rect;
import android.graphics.RectF;

import com.wilson_loo.traffic_label.remote.ClassifierRemote;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeOp.ResizeMethod;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.image.ops.Rot90Op;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

public abstract class Classifier {
    /**
     * Number of results to show in the UI.
     */
    private static final int MAX_RESULTS = 5;

    /**
     * The model type used for classification.
     */
    public enum Model {
        FLOAT_MOBILENET,
        QUANTIZED_MOBILENET,
        FLOAT_EFFICIENTNET,
        QUANTIZED_EFFICIENTNET,
        PYTHON_REMOTE,
        YOLO_V3,
    }

    /**
     * The runtime device type used for executing classification.
     */
    public enum Device {
        CPU,
        NNAPI,
        GPU
    }

    /**
     * The loaded TensorFlow Lite model.
     */
    private MappedByteBuffer tfliteModel;

    /**
     * Image size along the x axis.
     */
    protected int imageSizeX = 0;

    /**
     * Image size along the y axis.
     */
    protected int imageSizeY = 0;

    /**
     * An instance of the driver class to run model inference with Tensorflow Lite.
     */
    protected Interpreter tflite;

    /**
     * Options for configuring the Interpreter.
     */
    private final Interpreter.Options tfliteOptions = new Interpreter.Options();

    /**
     * Labels corresponding to the output of the vision model.
     */
    protected List<String> mLabelsList;

    /**
     * Input image TensorBuffer.
     */
    protected TensorImage mInputTensorImage;

    /**
     * Output probability TensorBuffer.
     */
    protected TensorBuffer mOutputProbabilityBuffer;

    /**
     * Processer to apply post processing of the output probability.
     */
    protected TensorProcessor mProbabilityProcessor;


    // 识别结果
    public static class Recognition {
        private final String id;
        private final String title;
        private final Float confidence;
        private final RectF location;

        public Recognition(final String id, final String title, final Float confidence, final RectF location) {
            this.id = id;
            this.title = title;
            this.confidence = confidence;
            this.location = location;
        }

        public String getTitle() {
            return this.title;
        }

        public String getName(){
            return this.title;
        }

        public Float getConfidence() {
            return this.confidence;
        }

        public Rect getRect(){
            return new Rect(
                    (int)this.location.left,
                    (int)this.location.top,
                    (int)this.location.right,
                    (int)this.location.bottom);
        }
    }

    // 内部构造函数，应该由工厂接口create创建实例
    protected Classifier(Activity activity, Device device, int numThreads, String modelPath) throws IOException {
        // 标签文件
        String labelPath = getLabelPath();
        mLabelsList = FileUtil.loadLabels(activity, labelPath);

        // 加载tflite 模型
        modelPath = modelPath == null ? getModelPath() : modelPath;
        if(modelPath == null){
            return;
        }

        tfliteModel = FileUtil.loadMappedFile(activity, modelPath);

        switch (device) {
            case NNAPI:
                NnApiDelegate nnApiDelegate = new NnApiDelegate();
                tfliteOptions.addDelegate(nnApiDelegate);
                break;
            case GPU:
                GpuDelegate gpuDelegate = new GpuDelegate();
                tfliteOptions.addDelegate(gpuDelegate);
                break;
            case CPU:
                break;
        }

        // 解析器的参数配置
        tfliteOptions.setNumThreads(numThreads);
        tflite = new Interpreter(tfliteModel, tfliteOptions);


        // 输入的图片尺寸信息
        {
            // 模型的输入数据结构信息
            int imageTensorIndex = 0;

            int[] inputShape = tflite.getInputTensor(imageTensorIndex).shape(); // {1, height, width, 3}
            imageSizeY = inputShape[1];
            imageSizeX = inputShape[2];

            // 输入的图片像素的元数据类型
            DataType inputDataType = tflite.getInputTensor(imageTensorIndex).dataType();

            // 构建输入图片的tensor
            mInputTensorImage = new TensorImage(inputDataType);
        }

        // 模型的输出数据结构信息（概率分布图）
        {
            int outputTensorIndex = 0;
            int[] outputShape = tflite.getOutputTensor(outputTensorIndex).shape(); // {1, NUM_CLASSES}
            DataType outputDataType = tflite.getOutputTensor(outputTensorIndex).dataType();

            // 模型输出的概率分布tensor
            mOutputProbabilityBuffer = TensorBuffer.createFixedSize(outputShape, outputDataType);
        }

        // 输出的概率分布图，需要进行归一化
        mProbabilityProcessor = new TensorProcessor.Builder().add(getPostprocessNormalizeOp()).build();
    }

    public List<Recognition> RecognizeImage(int faceId, final Bitmap bitmap, int sensorOrientation) {
        // 将原始图片载入成tensorflow 的图片张量
        mInputTensorImage.load(bitmap);
        int cropSize = Math.min(bitmap.getWidth(), bitmap.getHeight());
        int numRotation = sensorOrientation / 90;

        // TODO(b/143564309): Fuse ops inside ImageProcessor.
        ImageProcessor imageProcessor = new ImageProcessor.Builder()
            .add(new ResizeWithCropOrPadOp(cropSize, cropSize))
            .add(new ResizeOp(imageSizeX, imageSizeY, ResizeMethod.BILINEAR))
            .add(new Rot90Op(numRotation))
            .add(getPreprocessNormalizeOp())
            .build();

        TensorImage inputImageBuffer = imageProcessor.process(mInputTensorImage);

        // 进行实际的预测
        tflite.run(inputImageBuffer.getBuffer(), mOutputProbabilityBuffer.getBuffer().rewind());

        // 转换为 各个标签对应的概率
        Map<String/*标签*/, Float /*预测成该标签的概率*/> labeledProbability =
                new TensorLabel(mLabelsList, mProbabilityProcessor.process(mOutputProbabilityBuffer))
                        .getMapWithFloatValue();

        // 获取前几项结果
        return getTopKProbability(labeledProbability, 1);
    }

    // 获取前K排名
    private static List<Recognition> getTopKProbability(Map<String /*标签*/, Float /*预测成该标签的概率*/> labelProb, int topK) {
        topK = Math.min(topK, MAX_RESULTS);

        final ArrayList<Recognition> recognitionArrayList = new ArrayList<>();
        recognitionArrayList.clear();

        if (topK == 1) {
            // 只取第一个，使用快速算法
//            System.out.println("==============================");
            Map.Entry<String, Float> maxEntry = null;
            for (Map.Entry<String, Float> entry : labelProb.entrySet()) {
//                System.out.printf("%s : %.2f\n", entry.getKey(), entry.getValue());
                if (maxEntry == null || entry.getValue() > maxEntry.getValue()) {
                    maxEntry = entry;
                }
            }

            if (maxEntry != null) {
                recognitionArrayList.add(new Recognition("" + maxEntry.getKey(), maxEntry.getKey(), maxEntry.getValue(), null));
            }
        } else {
            // 使用最大堆实现
            PriorityQueue<Recognition> pq =
                    new PriorityQueue<>(
                            topK,
                            new Comparator<Recognition>() {
                                @Override
                                public int compare(Recognition o1, Recognition o2) {
                                    return o2.getConfidence().compareTo(o1.getConfidence());
                                }
                            });

            for (Map.Entry<String, Float> entry : labelProb.entrySet()) {
                pq.add(new Recognition("" + entry.getKey(), entry.getKey(), entry.getValue(), null));
            }

            int recognitionsSize = Math.min(pq.size(), topK);
            for (int k = 0; k < recognitionsSize; ++k) {
                recognitionArrayList.add(pq.poll());
            }
        }

        return recognitionArrayList;
    }

    // 获取tflite模型所在目录
    public abstract String getModelPath();

    // 获取分类标签所在目录
    public abstract String getLabelPath();

    /**
     * Gets the TensorOperator to nomalize the input image in preprocessing.
     */
    protected abstract TensorOperator getPreprocessNormalizeOp();

    /**
     * Gets the TensorOperator to dequantize the output probability in post processing.
     *
     * <p>For quantized model, we need de-quantize the prediction with NormalizeOp (as they are all
     * essentially linear transformation). For float model, de-quantize is not required. But to
     * uniform the API, de-quantize is added to float model too. Mean and std are set to 0.0f and
     * 1.0f, respectively.
     */
    protected abstract TensorOperator getPostprocessNormalizeOp();

    /**
     * 根据配置创建具体的 分类器 工厂方法设计模式： 只有一类抽象产品（abstract class Classifier），具体工厂（本匿名工厂）创建一个具体产品
     *
     * @param activity   The current Activity.
     * @param model      The model to use for classification.
     * @param device     The device to use for classification.
     * @param numThreads The number of threads to use for classification.
     * @return A classifier with the desired configuration.
     */
    public static Classifier Create(Activity activity, Model model, Device device, int numThreads, String modelPath)
            throws IOException {
        if (model == Model.FLOAT_MOBILENET) {
            return new ClassifierFloatMobileNet(activity, device, numThreads);
        }else if(model == Model.PYTHON_REMOTE) {
            return new ClassifierRemote(activity, device, numThreads, null);
        }else if(model == Model.YOLO_V3){
            return new ClassifierYoloV3(activity, device, numThreads, modelPath);
        } else {
            throw new UnsupportedOperationException("model:" + model);
        }
    }
}
