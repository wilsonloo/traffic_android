package com.arcsoft.trafficLabel.activity;


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

import com.arcsoft.arcfacedemo.R;
import com.arcsoft.trafficLabel.common.Constants;
import com.arcsoft.trafficLabel.model.DrawInfo;
import com.arcsoft.trafficLabel.tflite.Classifier;
import com.arcsoft.trafficLabel.util.DrawHelper;
import com.arcsoft.trafficLabel.util.camera.CameraHelper;
import com.arcsoft.trafficLabel.util.camera.CameraListener;
import com.arcsoft.trafficLabel.util.face.RecognizeColor;
import com.arcsoft.trafficLabel.widget.FaceRectView;
import com.arcsoft.face.AgeInfo;
import com.arcsoft.face.ErrorInfo;
import com.arcsoft.face.Face3DAngle;
import com.arcsoft.face.FaceEngine;
import com.arcsoft.face.FaceInfo;
import com.arcsoft.face.GenderInfo;
import com.arcsoft.face.LivenessInfo;
import com.tencent.bugly.crashreport.CrashReport;

import java.io.ByteArrayOutputStream;
import java.io.InvalidClassException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;


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
    private int processMask = FaceEngine.ASF_AGE | FaceEngine.ASF_FACE3DANGLE | FaceEngine.ASF_GENDER | FaceEngine.ASF_LIVENESS;


    /**
     * 相机预览显示的控件，可为SurfaceView或TextureView
     */
    private View previewView;
    private FaceRectView mTrafficLabelRectView;
    private FaceRectView emotionRectView;

    private Camera.Size previewSize;

    private CameraHelper cameraHelper;
    private DrawHelper drawHelper;

    private FaceEngine faceEngine;
    private int afCode = -1;

    private final Map<Integer, Bundle> mFaceLastEmotions = new HashMap<>();

    private Classifier mClassifier;
    public Classifier getClassifier(){ return mClassifier;}

    // 可以识别
    private boolean mTFEngineReady = true;

    // 私有函数列表 ***********************************************************
    private void initCamera() {
        DisplayMetrics metrics = new DisplayMetrics();
        getWindowManager().getDefaultDisplay().getMetrics(metrics);

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
                // 清空当前的所有框框
                if (mTrafficLabelRectView != null) {
                    mTrafficLabelRectView.clearFaceInfo();
                }

                if (!mTFEngineReady){
                    return;
                }

                // 摄像机画面对应的位图
                Bitmap previewBitmap = null;
                {
//                   camera.setOneShotPreviewCallback(null);
                    //处理data
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

                if (mTrafficLabelRectView != null && emotionRectView != null && drawHelper != null) {
                    Bitmap faceBitmap = previewBitmap;
                    Bundle bundle = null;

                    // 进行预测和绘制
                    Object[] predictResult = predict(0, faceBitmap);
                    if(predictResult != null) {
                        String emotionType = (String) predictResult[0];
                        Float confidence = (Float) predictResult[1];

                        bundle = new Bundle();
                        bundle.putString("emotionType", emotionType);
                        bundle.putFloat("confidence", confidence);
                        bundle.putInt("emotionResourceId", mClassifier.GetEmotionResourceId(emotionType));

                        mFaceLastEmotions.put(faceInfo.getFaceId(), bundle);
                    }else{
                        // load last emotion
                        bundle = mFaceLastEmotions.get(faceInfo.getFaceId());
                    }

                    DrawInfo newDrawInfo = new DrawInfo(
                            adjustFaceRect,
                            genderInfoList.get(i).getGender(),
                            ageInfoList.get(i).getAge(),
                            faceLivenessInfoList.get(i).getLiveness(),
                            RecognizeColor.COLOR_UNKNOWN,
                            null,
                            bundle);

                    drawInfoList.add(newDrawInfo);
                }
                drawHelper.draw(mTrafficLabelRectView, drawInfoList);
                drawHelper.draw(emotionRectView, drawInfoList);
            }

            // 进行预测
            private Object[] predict(int faceId, Bitmap faceBitmap) {
                // 进行分类预测，并产生表情
                Bitmap faceBitmap8888 = faceBitmap.copy(Bitmap.Config.ARGB_8888, true);

                ArrayList<Classifier.Recognition> recognitions = (ArrayList<Classifier.Recognition>) mClassifier.RecognizeImage(faceId, faceBitmap8888, 0);
                if (recognitions != null && recognitions.size() > 0) {
                    Classifier.Recognition predict = recognitions.get(0);
                    if (predict.getConfidence() > 0.8) {
                        // 获取表镜名称、对应图片资源
                        String emotionType = predict.getTitle();
                        Float confidence = predict.getConfidence();

                        return new Object[]{emotionType, confidence};
                    }
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

    private void initClassifier(int numThreads){
        try {
            if(mTensorflowType == Constants.TENSORFLOW_TYPE_FLASK_REMOTE) {
                mClassifier = Classifier.Create(this,
                        Classifier.Model.PYTHON_REMOTE,
                        Classifier.Device.GPU,
                        numThreads);
            }else if(mTensorflowType == Constants.TENSORFLOW_TYPE_TFLITE){
                mClassifier = Classifier.Create(this,
                        Classifier.Model.YOLO_V3,
                        Classifier.Device.GPU,
                        numThreads);
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

        // 分类器
        initClassifier(4);

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
