package com.wilson_loo.traffic_label.activity;


import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.ActivityInfo;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Point;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.hardware.Camera;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.support.v7.app.AppCompatActivity;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.View;
import android.view.ViewTreeObserver;
import android.view.WindowManager;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.tencent.bugly.crashreport.CrashReport;
import com.wilson_loo.traffic_label.R;
import com.wilson_loo.traffic_label.remote.ClassifierRemote;
import com.wilson_loo.traffic_label.tflite.Classifier;
import com.wilson_loo.traffic_label.tflite.ClassifierYoloV3;
import com.wilson_loo.traffic_label.util.BitmapUtils;
import com.wilson_loo.traffic_label.util.DrawHelper;
import com.wilson_loo.traffic_label.util.MediaDecoder;
import com.wilson_loo.traffic_label.util.OnGetBitmapListener;
import com.wilson_loo.traffic_label.util.camera.CameraHelper;
import com.wilson_loo.traffic_label.util.camera.CameraListener;
import com.wilson_loo.traffic_label.widget.FaceRectView;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InvalidClassException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class DetectTrafficLabelStaticActivity extends AppCompatActivity implements ViewTreeObserver.OnGlobalLayoutListener{

    private class OnGetBitmapImpl implements OnGetBitmapListener {
        private final DetectTrafficLabelStaticActivity mAct;

        public OnGetBitmapImpl(DetectTrafficLabelStaticActivity act){
            mAct = act;
        }

        @Override
        public void getBitmap(Bitmap bitmap, long timeMs) {
            mAct.mBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);;
            mAct.doShowPicture();
        }
    };

    /**
     * 所需的所有权限信息
     */
    private static final int ACTION_REQUEST_PERMISSIONS = 0x001;
    private static final String[] NEEDED_PERMISSIONS = new String[]{
            Manifest.permission.CAMERA,
            Manifest.permission.READ_PHONE_STATE
    };

    private static final String TAG = "DetectTrafficStatic";
    private Integer rgbCameraId = Camera.CameraInfo.CAMERA_FACING_BACK;
    private String mTensorflowType = null;

    public static final int DETECT_STATE_FREE = 0;
    public static final int DETECT_STATE_DETECTING = 1;
    public static final int DETECT_STATE_DRAWING = 2;

    private static int WRITE_SD_CODE = 1;
    private static int READ_SD_CODE = 2;

    private static final int PICK_IMAGE = 1;
    private static final int PICK_VIDEO = 2;

    /**
     * 相机预览显示的控件，可为SurfaceView或TextureView
     */
    private View mPreviewView;
    private FaceRectView mTrafficLabelRectView;

    private Camera.Size previewSize;

    private CameraHelper cameraHelper;
    private DrawHelper drawHelper;

    private ImageView mImageView;
    private Button mButton;
    private Bitmap mBitmap;

    private int afCode = -1;
    private final int rectLineSize = 3;
    private final int videoCaptureInterval = 100;

    private final Map<Integer, Bundle> mFaceLastEmotions = new HashMap<>();

    private int mDetectState = DETECT_STATE_FREE;
    public int getDetectState(){ return mDetectState;}
    public void setDetectState(int state){ mDetectState = state;}
    private final ExecutorService mSingleThreadExecutorForDetect = Executors.newSingleThreadExecutor();

    private TextView mTextViewDetectResult = null;

    private ArrayList<Classifier.Recognition> mLastRecognitions = null;

    private Thread mVideoThread = null;
    private MediaDecoder mMediaDecoder = null;
    private int mVideoMillSecondsLength = 0;
    private int mVideoCurrent = 0;

    private Classifier mClassifier;
    public Classifier getClassifier(){ return mClassifier;}

    private OnGetBitmapImpl mOnGetBitmapImpl = new OnGetBitmapImpl(this);

    // 私有函数列表 ***********************************************************
    private void initClassifier(float scoreThreshold, int numThreads){
        try {
            if(mTensorflowType.equals("remote flask")) {
                mClassifier = Classifier.Create(this,
                        Classifier.Model.PYTHON_REMOTE,
                        Classifier.Device.CPU,
                        numThreads,
                        null);

            }else{
                mClassifier = Classifier.Create(this,
                        Classifier.Model.YOLO_V3,
                        Classifier.Device.CPU,
                        numThreads,
                        mTensorflowType);
                ((ClassifierYoloV3)mClassifier).setScoreThreshold(scoreThreshold);
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
        mTensorflowType = intent.getStringExtra("tensorflowType");
        float scoreThreshold = intent.getFloatExtra("scoreThreshold", 0.3f);

        setContentView(R.layout.activity_detect_traffic_static_signal);
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

        mImageView = findViewById(R.id.static_traffic_pic);

        findViewById(R.id.btnLoadVideo).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
//                loadDetectVideo();
                if(Build.VERSION.SDK_INT >= Build.VERSION_CODES.KITKAT){
                    Intent intent = new Intent(Intent.ACTION_PICK);
                    intent.setType("video/*");
                    startActivityForResult(intent, PICK_VIDEO);
                }else{
                    assert false;
                }
            }
        });

        findViewById(R.id.btnLoadImage_1).setOnClickListener(  new View.OnClickListener(){
            @Override
            public void onClick(View view) {
                showPicture("speed_limit_60.jpg");
            }
        });
        findViewById(R.id.btnLoadImage_2).setOnClickListener(  new View.OnClickListener(){
            @Override
            public void onClick(View view) {
                showPicture("speed_limit_50_no_laba.jpg");
            }
        });
        findViewById(R.id.btnLoadImage_3).setOnClickListener(  new View.OnClickListener(){
            @Override
            public void onClick(View view) {
                showPicture("real_traffic_screen_shot.png");
            }
        });

        findViewById(R.id.btn_load_and_detect).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if(Build.VERSION.SDK_INT >= Build.VERSION_CODES.KITKAT){
                    Intent intent = new Intent(Intent.ACTION_PICK);
                    intent.setType("image/*");
                    startActivityForResult(intent, PICK_IMAGE);
                }else{
                    assert false;
                }
            }
        });

        // 分类器
        initClassifier(scoreThreshold, 16);

        //在布局结束后才做初始化操
        mPreviewView.getViewTreeObserver().addOnGlobalLayoutListener(this);
    }

    public void onActivityResult(int requestCode, int resultCode, Intent data){
        super.onActivityResult(requestCode, resultCode, data);

        if(resultCode == Activity.RESULT_OK){
            if(requestCode == PICK_IMAGE) {
                if(Build.VERSION.SDK_INT >= Build.VERSION_CODES.KITKAT) {
                    Cursor cursor = this.getContentResolver().query(data.getData(),
                            null, null, null, null);

                    cursor.moveToFirst();
                    int idx = cursor.getColumnIndex(MediaStore.Images.ImageColumns.DATA);
                    String fileSrc = cursor.getString(idx);
                    cursor.close();

                    mBitmap = BitmapFactory.decodeFile(fileSrc);
                    doShowPicture();
                }else{
                    assert false;
                }
            }else if(requestCode == PICK_VIDEO){
                if(Build.VERSION.SDK_INT >= Build.VERSION_CODES.KITKAT) {
                    Cursor cursor = this.getContentResolver().query(data.getData(),
                            null, null, null, null);

                    cursor.moveToFirst();
                    int idx = cursor.getColumnIndex(MediaStore.Images.ImageColumns.DATA);
                    String fileSrc = cursor.getString(idx);
                    cursor.close();

                    loadDetectVideo(fileSrc);
                }else{
                    assert false;
                }
            }
        }
    }

    /**
     * 在{@link #mPreviewView}第一次布局完成后，去除该监听，并且进行引擎和相机的初始化
     */
    @Override
    public void onGlobalLayout() {
        mPreviewView.getViewTreeObserver().removeOnGlobalLayoutListener(this);
        initCamera();
    }

    private void initCamera() {
        DisplayMetrics metrics = new DisplayMetrics(); getWindowManager().getDefaultDisplay().getMetrics(metrics);

        AppCompatActivity activity = this;

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
                if (mDetectState == DETECT_STATE_FREE) {
                    if (mVideoCurrent + videoCaptureInterval < mVideoMillSecondsLength) {

                        mVideoCurrent += videoCaptureInterval;
                        mMediaDecoder.decodeFrame(mVideoCurrent, mOnGetBitmapImpl);
                    }
                }
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

    private void loadDetectVideo(String filePath){
        Log.d("LWS", "load video..."+filePath);

//        String filePath = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_MOVIES) + "/" + fileName;
        mMediaDecoder = new MediaDecoder(filePath);

        mVideoMillSecondsLength = Integer.parseInt(mMediaDecoder.getVedioFileLength());
        mVideoCurrent = 0;
    }

    private void showPicture(String imageName){

        try {
            Log.d("LWS", "load image...");
            InputStream is = this.getAssets().open(imageName);
            mBitmap = BitmapFactory.decodeStream(is);
            doShowPicture();

        }catch (IOException e){
            e.printStackTrace();;
        }
    }

    private void doShowPicture() {
        Bitmap tempBitmap = mBitmap.copy(Bitmap.Config.ARGB_8888, true);
        Canvas canvas = new Canvas(tempBitmap);
        Log.d("LWS", "load image...done");

        // 进行预测
        Log.d("LWS", "predict...");
        ArrayList<Classifier.Recognition> recognitions = predict(0, mBitmap);
        Log.d("LWS", "predict...done");

        doDrawRecognitions(tempBitmap, canvas, recognitions);
    }

    private void doDrawRecognitions(Bitmap tempBitmap, Canvas canvas, ArrayList<Classifier.Recognition> recognitions) {
        // 绘制识别结果
        Log.d("LWS", "draw result...");
        if(recognitions != null) {
            for (Iterator iter = recognitions.iterator(); iter.hasNext(); ) {
                Classifier.Recognition box = (Classifier.Recognition) iter.next();
                Paint paint = new Paint();
                paint.setColor(Color.RED);
                paint.setTextSize(24);
                canvas.drawText(box.getTitle(), box.getRect().left, box.getRect().top - rectLineSize, paint);

                paint.setStyle(Paint.Style.STROKE);
                paint.setStrokeWidth(rectLineSize);
                canvas.drawRect(box.getRect().left, box.getRect().top, box.getRect().right, box.getRect().bottom, paint);
            }
        }
        mImageView.setImageBitmap(tempBitmap);
        Log.d("LWS", "draw result...done");
    }

    // 进行预测
    private ArrayList<Classifier.Recognition> predict(int faceId, Bitmap faceBitmap) {
        // 进行分类预测，并产生表情
        Bitmap faceBitmap8888 = faceBitmap.copy(Bitmap.Config.ARGB_8888, true);

        ArrayList<Classifier.Recognition> recognitions = (ArrayList<Classifier.Recognition>) mClassifier.RecognizeImage(faceId, faceBitmap8888, 0);
        if (recognitions != null && recognitions.size() > 0) {
            String result = "";
            for(int k = 0; k < recognitions.size(); ++k){
                result += recognitions.get(k).getName()+" : " + recognitions.get(k).getConfidence() + "\n";
            }
            Log.e("LWS", ">>>>>>>>>>>>>>>>>>>> \n" + result);

            return recognitions;
        }

        return null;
    }

    private void addRecognitions(ArrayList<Classifier.Recognition> recognitions){

    }
}
