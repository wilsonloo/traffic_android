package com.wilson_loo.traffic_label.activity;


import android.Manifest;
import android.content.Intent;
import android.content.pm.ActivityInfo;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Point;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.hardware.Camera;
import android.os.Build;
import android.os.Bundle;
import android.support.v4.app.ActivityCompat;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.View;
import android.view.ViewTreeObserver;
import android.view.WindowManager;
import android.widget.TextView;

import com.arcsoft.trafficLabel.common.Constants;
import com.arcsoft.trafficLabel.model.DrawInfo;
import com.wilson_loo.traffic_label.R;
import com.wilson_loo.traffic_label.tflite.ClassifierYoloV3;
import com.wilson_loo.traffic_label.util.DrawHelper;
import com.wilson_loo.traffic_label.util.camera.CameraHelper;
import com.wilson_loo.traffic_label.util.camera.CameraHelper;
import com.wilson_loo.traffic_label.util.camera.CameraListener;
import com.tencent.bugly.crashreport.CrashReport;
import com.wilson_loo.traffic_label.tflite.Classifier;
import com.wilson_loo.traffic_label.widget.FaceRectView;

import java.io.ByteArrayOutputStream;
import java.io.InvalidClassException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;


public class DetectTrafficLabelActivity extends BaseActivity implements ViewTreeObserver.OnGlobalLayoutListener {
    /**
     * 所需的所有权限信息
     */
    private static final int ACTION_REQUEST_PERMISSIONS = 0x001;
    private static final String[] NEEDED_PERMISSIONS = new String[]{
            Manifest.permission.CAMERA,
            Manifest.permission.READ_PHONE_STATE
    };

    private static final String TAG = "DetectTrafficActivity";
    private Integer rgbCameraId = Camera.CameraInfo.CAMERA_FACING_BACK;
    private Integer mTensorflowType = Constants.TENSORFLOW_TYPE_TFLITE;

    public static final int DETECT_STATE_FREE = 0;
    public static final int DETECT_STATE_DETECTING = 1;
    public static final int DETECT_STATE_DRAWING = 2;

    /**
     * 相机预览显示的控件，可为SurfaceView或TextureView
     */
    private View previewView;
    private FaceRectView mTrafficLabelRectView;

    private Camera.Size previewSize;

    private CameraHelper cameraHelper;
    private DrawHelper drawHelper;

    private int afCode = -1;

    private final Map<Integer, Bundle> mFaceLastEmotions = new HashMap<>();

    private int mDetectState = DETECT_STATE_FREE;
    public int getDetectState(){ return mDetectState;}
    public void setDetectState(int state){ mDetectState = state;}
    private final ExecutorService mSingleThreadExecutorForDetect = Executors.newSingleThreadExecutor();

    private TextView mTextViewDetectResult = null;

    private ArrayList<Classifier.Recognition> mLastRecognitions = null;

    private Classifier mClassifier;
    public Classifier getClassifier(){ return mClassifier;}

    // 私有函数列表 ***********************************************************
    private void initCamera() {
        DisplayMetrics metrics = new DisplayMetrics(); getWindowManager().getDefaultDisplay().getMetrics(metrics);

        CameraListener cameraListener = new CameraListener() {
            @Override
            public void onCameraOpened(Camera camera, int cameraId, int displayOrientation, boolean isMirror) {
                Log.i(TAG, "onCameraOpened: " + cameraId + "  " + displayOrientation + " " + isMirror);
                previewSize = camera.getParameters().getPreviewSize();
                drawHelper = new DrawHelper(previewSize.width,
                        previewSize.height,
                        previewView.getWidth(),
                        previewView.getHeight(),
                        displayOrientation,
                        cameraId,
                        isMirror,
                        false,
                        false);
            }

            @Override
            public void onPreview(byte[] nv21, Camera camera) {
                if(mDetectState == DETECT_STATE_DRAWING) {
                    // 有结果了，需要进行绘制
                    if (mLastRecognitions != null) {
                        // 清空当前的所有框框
                        if (mTrafficLabelRectView != null) {
                            mTrafficLabelRectView.clearFaceInfo();
                        }

                        drawHelper.draw(mTrafficLabelRectView, mLastRecognitions);
                        mLastRecognitions = null;
                    }

                    // 在下一帧开始检测
                    mDetectState = DETECT_STATE_FREE;
                    return;

                }else if(mDetectState == DETECT_STATE_DETECTING){
                    // 还在检测中
                    return;

                }else{
                    // 可以开始检测了
                    mDetectState = DETECT_STATE_DETECTING;

                    // 摄像机画面对应的位图
                    Bitmap previewBitmap = null;
                    {
                        // camera.setOneShotPreviewCallback(null);
                        // 处理data
                        Camera.Size previewSize = camera.getParameters().getPreviewSize();//获取尺寸,格式转换的时候要用到
                        BitmapFactory.Options newOpts = new BitmapFactory.Options();
                        newOpts.inJustDecodeBounds = true;
                        YuvImage yuvimage = new YuvImage(
                                nv21,
                                ImageFormat.NV21,
                                previewSize.width,
                                previewSize.height,
                                null);
                        ByteArrayOutputStream baos = new ByteArrayOutputStream();
                        yuvimage.compressToJpeg(new Rect(0, 0, previewSize.width, previewSize.height), 100, baos);// 80--JPG图片的质量[0-100],100最高
                        byte[] rawImage = baos.toByteArray();
                        //将rawImage转换成bitmap
                        BitmapFactory.Options options = new BitmapFactory.Options();
                        options.inPreferredConfig = Bitmap.Config.RGB_565;
                        previewBitmap = BitmapFactory.decodeByteArray(rawImage, 0, rawImage.length, options);
                    }

                    Bitmap finalPreviewBitmap = previewBitmap;
                    mSingleThreadExecutorForDetect.execute(new Runnable() {
                        @Override
                        public void run() {
                            if (mTrafficLabelRectView != null && drawHelper != null) {
                                Bitmap faceBitmap = finalPreviewBitmap;
                                Bundle bundle = null;

                                // 进行预测和绘制
                                Object[] predictResult = predict(0, faceBitmap);
                                if (predictResult != null) {
                                    mLastRecognitions = (ArrayList<Classifier.Recognition>) predictResult[0];
//                                    drawHelper.draw(mTrafficLabelRectView, recognitions);
                                    mDetectState = DETECT_STATE_DRAWING;
                                }else{
                                    mDetectState = DETECT_STATE_FREE;
                                    mLastRecognitions = null;
                                }
                            }
                        }
                    });
                }
            }

            // 进行预测
            private Object[] predict(int faceId, Bitmap faceBitmap) {
                // 进行分类预测，并产生表情
                Bitmap faceBitmap8888 = faceBitmap.copy(Bitmap.Config.ARGB_8888, true);

                ArrayList<Classifier.Recognition> recognitions = (ArrayList<Classifier.Recognition>) mClassifier.RecognizeImage(faceId, faceBitmap8888, 0);
                if (recognitions != null && recognitions.size() > 0) {
                    String result = "";
                    for(int k = 0; k < recognitions.size(); ++k){
                        result += recognitions.get(k).getName()+" : " + recognitions.get(k).getConfidence() + "\n";
                    }
                    Log.e("= Detect result", ">>>>>>>>>>>>>>>>>>>> " + result);

                    return new Object[]{recognitions};
                }

                return null;
            }

            @Override
            public void onCameraClosed() {
                Log.i(TAG, "onCameraClosed: ");
            }

            @Override
            public void onCameraError(Exception e) {
                Log.i(TAG, "onCameraError: " + e.getMessage());
            }

            @Override
            public void onCameraConfigurationChanged(int cameraID, int displayOrientation) {
                if (drawHelper != null) {
                    drawHelper.setCameraDisplayOrientation(displayOrientation);
                }
                Log.i(TAG, "onCameraConfigurationChanged: " + cameraID + "  " + displayOrientation);
            }
        };

        cameraHelper = new CameraHelper.Builder()
                .previewViewSize(new Point(previewView.getMeasuredWidth(), previewView.getMeasuredHeight()))
                .rotation(getWindowManager().getDefaultDisplay().getRotation())
                .specificCameraId(rgbCameraId != null ? rgbCameraId : Camera.CameraInfo.CAMERA_FACING_FRONT)
                .isMirror(false)
                .previewOn(previewView)
                .cameraListener(cameraListener)
                .build();
        cameraHelper.init();
        cameraHelper.start();
    }

    private void initClassifier(float scoreThreshold, int numThreads){
        try {
            if(mTensorflowType == Constants.TENSORFLOW_TYPE_FLASK_REMOTE) {
                mClassifier = Classifier.Create(this,
                        Classifier.Model.PYTHON_REMOTE,
                        Classifier.Device.CPU,
                        numThreads);
            }else if(mTensorflowType == Constants.TENSORFLOW_TYPE_TFLITE){
                mClassifier = Classifier.Create(this,
                        Classifier.Model.YOLO_V3,
                        Classifier.Device.CPU,
                        numThreads);
                ((ClassifierYoloV3)mClassifier).setScoreThreshold(scoreThreshold);
            }else{
                throw new InvalidClassException("tensorflow usage type, flaskremote or tflite");
            }

        } catch (Exception e) {
            e.printStackTrace();
            CrashReport.postCatchedException(e);
        }
    }

    // 函数列表 ***********************************************************
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // 摄像头
        Intent intent = getIntent();
        rgbCameraId = intent.getIntExtra("whichCamera", Camera.CameraInfo.CAMERA_FACING_BACK);
        mTensorflowType = intent.getIntExtra("tensorflowType", Constants.TENSORFLOW_TYPE_TFLITE);
        float scoreThreshold = intent.getFloatExtra("scoreThreshold", 0.3f);

        setContentView(R.layout.activity_detect_traffic_signal);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.KITKAT) {
            WindowManager.LayoutParams attributes = getWindow().getAttributes();
            attributes.systemUiVisibility = View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY | View.SYSTEM_UI_FLAG_HIDE_NAVIGATION;
            getWindow().setAttributes(attributes);
        }

        // Activity启动后就锁定为启动时的方向
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LOCKED);

        previewView = findViewById(R.id.fe_texture_preview);
        mTrafficLabelRectView = findViewById(R.id.fe_traffic_label_rect_view);
        mTextViewDetectResult = findViewById(R.id.textDetectResult);

        // 分类器
        initClassifier(scoreThreshold, 4);

        //在布局结束后才做初始化操作
        previewView.getViewTreeObserver().addOnGlobalLayoutListener(this);
    }

    @Override
    void afterRequestPermission(int requestCode, boolean isAllGranted) {

    }

    /**
     * 在{@link #previewView}第一次布局完成后，去除该监听，并且进行引擎和相机的初始化
     */
    @Override
    public void onGlobalLayout() {
        previewView.getViewTreeObserver().removeOnGlobalLayoutListener(this);
        if (!checkPermissions(NEEDED_PERMISSIONS)) {
            ActivityCompat.requestPermissions(this, NEEDED_PERMISSIONS, ACTION_REQUEST_PERMISSIONS);
        } else {
            initCamera();
        }
    }
}