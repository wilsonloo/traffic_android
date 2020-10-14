package com.wilson_loo.traffic_label.remote;

import android.app.Activity;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.RectF;

import com.wilson_loo.traffic_label.util.BitmapUtils;
import com.wilson_loo.traffic_label.tflite.Classifier;

import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.ops.NormalizeOp;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

public class ClassifierRemote extends Classifier {

    /** Float MobileNet requires additional normalization of the used input. */
    private static final float IMAGE_MEAN = 0.0f;

    private static final float IMAGE_STD = 255.0f;

    /**
     * Float model does not need dequantization in the post-processing. Setting mean and std as 0.0f
     * and 1.0f, repectively, to bypass the normalization.
     */
    private static final float PROBABILITY_MEAN = 0.0f;

    private static final float PROBABILITY_STD = 1.0f;

    private Activity mActivity = null;
    private Thread mRpcThread = null;
    private final ArrayList<Recognition> recognitionArrayList = new ArrayList<>();
    private static Lock mLocker = new ReentrantLock();

    public ClassifierRemote(Activity activity, Device device, int numThreads, String modelPath)
            throws IOException
    {
        super(activity, device, numThreads, modelPath);

        mActivity = activity;
    }

    @Override
    public List<Classifier.Recognition> RecognizeImage(int faceId, final Bitmap bitmap, int sensorOrientation) {
        mLocker.lock();
        {
            if(mRpcThread != null && mRpcThread.isAlive()){
                mLocker.unlock();
                return null;
            }

            if(!recognitionArrayList.isEmpty()){
                ArrayList<Recognition> retList = (ArrayList<Recognition>) recognitionArrayList.clone();
                recognitionArrayList.clear();
                mLocker.unlock();
                return retList;
            }
        }
        mLocker.unlock();

        mRpcThread = doAsyncRpc(bitmap);

        return null;
    }

    private Thread doAsyncRpc(Bitmap bitmap){
        Thread rpcThread = new Thread(new Runnable() {
            @Override
            public void run() {
                AssetManager assetManager = mActivity.getAssets();
                try {
                    System.out.println("=====================> begin to rpc call");
//                    InputStream is = assetManager.open("happy_11.jpg");    //直接写assets文件夹下的图片名就可以
//                    InputStream is = new FileInputStream(imagePath);
//                    ByteArrayOutputStream baos = new ByteArrayOutputStream();
//                    int k = -1;
//                    while ((k = is.read()) != -1) {
//                        baos.write(k);
//                    }
//                    byte[] imageBinary = baos.toByteArray();
//                    is.close();

                    //Bitmap转换成byte[]
                    ByteArrayOutputStream baos = new ByteArrayOutputStream();
                    bitmap.compress(Bitmap.CompressFormat.JPEG, 100, baos);
                    byte[] imageBinary = baos.toByteArray();

                    String path = "http://192.168.1.19:5000/classify_traffic?binary=1";
                    URL url = new URL(path);
                    HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                    conn.setConnectTimeout(5000);
                    conn.setReadTimeout(5000);
                    conn.setDoInput(true);
                    conn.setDoOutput(true);
                    conn.setRequestMethod("POST");
                    conn.setUseCaches(false);
                    conn.setRequestProperty("Content-Type", "application/x-www-form-urlencoded");

//                    conn.setRequestProperty("Content-Length", String.valueOf(data.length()));
                    conn.setRequestProperty(
                            "User-Agent",
                            "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.101 Safari/537.36");

                    conn.getOutputStream().write(imageBinary);

                    int code = conn.getResponseCode();
                    if (code == 200) {
                        InputStream response = conn.getInputStream();
                        ByteArrayOutputStream bos = new ByteArrayOutputStream();
                        byte[] buffer = new byte[1024];
                        int len = -1;
                        while ((len = response.read(buffer)) != -1) {
                            bos.write(buffer, 0, len);
                        }
                        String result = new String(bos.toByteArray());
                        if(result != null && !result.isEmpty()) {
                            String boxes[] = result.split("\\|");

                            mLocker.lock();
                            for (int k = 0; k < boxes.length; ++k) {
                                String elems[] = boxes[k].split(":");
                                float left = Float.valueOf(elems[0]);
                                float top = Float.valueOf(elems[1]);
                                float right = Float.valueOf(elems[2]);
                                float bottom = Float.valueOf(elems[3]);
                                float prob = Float.valueOf(elems[4]);
                                float label = Float.valueOf(elems[5]);
                                RectF rect = new RectF(left, top, right, bottom);
                                String title = mLabelsList.get((int) label);
                                Recognition recognition = new Recognition(title, title, prob, rect);
                                recognitionArrayList.add(recognition);
                            }
                            mLocker.unlock();
                        }

//                        baos.close();
                        response.close();
                        conn.disconnect();
                    } else {
                        System.out.printf("call with error mesage, code:%d\n", code);
                    }
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        });

        rpcThread.start();
        return rpcThread;
    }

    @Override
    public String getModelPath() {
        return null;
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
