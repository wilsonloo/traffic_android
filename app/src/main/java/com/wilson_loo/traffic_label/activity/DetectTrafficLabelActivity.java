package com.wilson_loo.traffic_label.activity;


import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.pm.ActivityInfo;
import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Point;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.hardware.Camera;
import android.icu.text.UFormat;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.support.v4.app.ActivityCompat;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.View;
import android.view.ViewTreeObserver;
import android.view.WindowManager;
import android.widget.TextView;
import android.widget.Toast;

import com.arcsoft.trafficLabel.common.Constants;
import com.wilson_loo.traffic_label.R;
import com.wilson_loo.traffic_label.tflite.ClassifierYoloV3;
import com.wilson_loo.traffic_label.util.BitmapUtils;
import com.wilson_loo.traffic_label.util.DrawHelper;
import com.wilson_loo.traffic_label.util.camera.CameraHelper;
import com.wilson_loo.traffic_label.util.camera.CameraListener;
import com.tencent.bugly.crashreport.CrashReport;
import com.wilson_loo.traffic_label.tflite.Classifier;
import com.wilson_loo.traffic_label.widget.FaceRectView;

import java.io.BufferedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InvalidClassException;
import java.io.OutputStream;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
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
            Manifest.permission.READ_PHONE_STATE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE,
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
    private View mPreviewView;
    private FaceRectView mTrafficLabelRectView;

    private Camera.Size previewSize;

    private CameraHelper cameraHelper;
    private DrawHelper drawHelper;

    private int afCode = -1;
    private boolean mToScreenShot = false;

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

        BaseActivity activity = this;

        CameraListener cameraListener = new CameraListener() {
            @Override
            public void onCameraOpened(Camera camera, int cameraId, int displayOrientation, boolean isMirror) {
                Log.i(TAG, "onCameraOpened: " + cameraId + "  " + displayOrientation + " " + isMirror);
                previewSize = camera.getParameters().getPreviewSize();
                drawHelper = new DrawHelper(previewSize.width,
                        previewSize.height,
                        mPreviewView.getWidth(),
                        mPreviewView.getHeight(),
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

                    Camera.Size previewSize = camera.getParameters().getPreviewSize();//获取尺寸,格式转换的时候要用到

                    // 摄像机画面对应的位图
                    Bitmap previewBitmap = null;
                    if(false)
                    {
                        // camera.setOneShotPreviewCallback(null);
                        // 处理data
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
                        options.inPreferredConfig = Bitmap.Config.ARGB_8888;
                        previewBitmap = BitmapFactory.decodeByteArray(rawImage, 0, rawImage.length, options);
                    }
                    if(true){
                        int[] rgbBytes = new int[previewSize.width * previewSize.height];
                        BitmapUtils.convertYUV420SPToARGB8888(nv21, previewSize.width, previewSize.height, rgbBytes);

                        Bitmap rgbFrameBitmap = Bitmap.createBitmap(previewSize.width, previewSize.height, Bitmap.Config.ARGB_8888);
                        rgbFrameBitmap.setPixels(rgbBytes, 0, previewSize.width, 0, 0, previewSize.width, previewSize.height);
                        previewBitmap = rgbFrameBitmap;// BitmapUtils.adjustToSize(rgbFrameBitmap, 800, 800);
                    }

                    Matrix matrix = new Matrix();
                    matrix.postRotate(90);
                    Bitmap finalPreviewBitmap = Bitmap.createBitmap(previewBitmap, 0, 0, previewBitmap.getWidth(), previewBitmap.getHeight(), matrix, true);

                    if(mToScreenShot){
                        saveScreenShot(finalPreviewBitmap);
                    }

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
                    Log.e("LWS", ">>>>>>>>>>>>>>>>>>>> " + result);

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
                .previewViewSize(new Point(mPreviewView.getMeasuredWidth(), mPreviewView.getMeasuredHeight()))
                .rotation(getWindowManager().getDefaultDisplay().getRotation())
                .specificCameraId(rgbCameraId != null ? rgbCameraId : Camera.CameraInfo.CAMERA_FACING_FRONT)
                .isMirror(false)
                .previewOn(mPreviewView)
                .cameraListener(cameraListener)
                .build();
        cameraHelper.init();
        cameraHelper.start();
    }

    private void saveScreenShot(Bitmap finalPreviewBitmap) {
        mToScreenShot = false;
        Date day=new Date();
        SimpleDateFormat df = new SimpleDateFormat("yyyy-MM-dd_HH_mm_ss");
        String fileName = "traffic_"+df.format(day)+".jpg";
        File file = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES), fileName);
        try{
            if(!file.exists()){
                file.createNewFile();
            }

            boolean ret = save(finalPreviewBitmap, file, Bitmap.CompressFormat.JPEG, false);
            if(ret){
                Toast.makeText(getApplicationContext(), "saved to " + file.getAbsolutePath(), Toast.LENGTH_SHORT).show();
            }

        }catch (Exception e){
            e.printStackTrace();
        }
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

        mPreviewView = findViewById(R.id.fe_texture_preview);
        mTrafficLabelRectView = findViewById(R.id.fe_traffic_label_rect_view);
        mTextViewDetectResult = findViewById(R.id.textDetectResult);

        // 分类器
        initClassifier(scoreThreshold, 8);

        //在布局结束后才做初始化操作
        mPreviewView.getViewTreeObserver().addOnGlobalLayoutListener(this);
    }

    @Override
    void afterRequestPermission(int requestCode, boolean isAllGranted) {

    }

    /**
     * 在{@link #mPreviewView}第一次布局完成后，去除该监听，并且进行引擎和相机的初始化
     */
    @Override
    public void onGlobalLayout() {
        mPreviewView.getViewTreeObserver().removeOnGlobalLayoutListener(this);
        if (!checkPermissions(NEEDED_PERMISSIONS)) {
            ActivityCompat.requestPermissions(this, NEEDED_PERMISSIONS, ACTION_REQUEST_PERMISSIONS);
        } else {
            initCamera();
            findViewById(R.id.btn_screenshoot_traffic_signal).setOnClickListener(
                    new View.OnClickListener(){
                        @Override
                        public void onClick(View view) {
                            mToScreenShot = true;
                        }
                    }
            );
        }
    }

    public static boolean save(Bitmap src, File file, Bitmap.CompressFormat format, boolean recycle){
        if(isEmptyBitmap(src)){
            return false;
        }

        OutputStream os;
        boolean ret = false;
        try{
            os = new BufferedOutputStream(new FileOutputStream(file));
            ret = src.compress(format, 100, os);
            if(recycle && !src.isRecycled()){
                src.recycle();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return ret;
    }

    public static boolean isEmptyBitmap(Bitmap src){
        return src == null || src.getWidth() == 0 || src.getHeight() == 0;
    }
}
