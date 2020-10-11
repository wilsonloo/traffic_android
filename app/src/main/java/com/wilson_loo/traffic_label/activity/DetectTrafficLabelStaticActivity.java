package com.wilson_loo.traffic_label.activity;


import android.Manifest;
import android.annotation.TargetApi;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.Paint;
import android.graphics.Point;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.hardware.Camera;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.Trace;
import android.provider.DocumentsContract;
import android.provider.MediaStore;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.View;
import android.view.ViewTreeObserver;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.arcsoft.trafficLabel.common.Constants;
import com.tencent.bugly.crashreport.CrashReport;
import com.wilson_loo.traffic_label.R;
import com.wilson_loo.traffic_label.tflite.Classifier;
import com.wilson_loo.traffic_label.tflite.ClassifierYoloV3;
import com.wilson_loo.traffic_label.util.DrawHelper;
import com.wilson_loo.traffic_label.util.MediaDecoder;
import com.wilson_loo.traffic_label.util.OnGetBitmapListener;
import com.wilson_loo.traffic_label.util.camera.CameraHelper;
import com.wilson_loo.traffic_label.util.camera.CameraListener;
import com.wilson_loo.traffic_label.widget.FaceRectView;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InvalidClassException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Semaphore;

public class DetectTrafficLabelStaticActivity extends AppCompatActivity implements ViewTreeObserver.OnGlobalLayoutListener, Camera.PreviewCallback {


    private class OnGetBitmapImpl implements OnGetBitmapListener {
        private final DetectTrafficLabelStaticActivity mAct;

        public OnGetBitmapImpl(DetectTrafficLabelStaticActivity act){
            mAct = act;
        }

        @Override
        public void getBitmap(Bitmap bitmap, long timeMs) {
            mAct.mBitmap = bitmap;
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
    private Integer mTensorflowType = Constants.TENSORFLOW_TYPE_TFLITE;

    public static final int DETECT_STATE_FREE = 0;
    public static final int DETECT_STATE_DETECTING = 1;
    public static final int DETECT_STATE_DRAWING = 2;

    private static int WRITE_SD_CODE = 1;
    private static int READ_SD_CODE = 2;

    /**
     * 相机预览显示的控件，可为SurfaceView或TextureView
     */
    private View previewView;
    private FaceRectView mTrafficLabelRectView;

    private Camera.Size previewSize;

    private CameraHelper cameraHelper;
    private DrawHelper drawHelper;

    private ImageView mImageView;
    private Button mButton;
    private Bitmap mBitmap;

    private int afCode = -1;
    private final int rectLineSize = 3;

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

        setContentView(R.layout.activity_detect_traffic_static_signal);
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

        mImageView = findViewById(R.id.static_traffic_pic);

        findViewById(R.id.btnLoadVideo).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                loadDetectVideo();
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

        findViewById(R.id.btn_load_and_detect).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if(Build.VERSION.SDK_INT >= Build.VERSION_CODES.KITKAT){
                    Intent intent = new Intent(Intent.ACTION_PICK);
                    intent.setType("image/*");
                    startActivityForResult(intent, 1);
                }else{
                    assert false;
                }
            }
        });

        // 分类器
        initClassifier(scoreThreshold, 4);

        //在布局结束后才做初始化操作
//        previewView.getViewTreeObserver().addOnGlobalLayoutListener(this);
    }

    public void onActivityResult(int requestCode, int resultCode, Intent data){
        super.onActivityResult(requestCode, resultCode, data);

        if(resultCode == Activity.RESULT_OK){
            if(requestCode == 1) {
                if(Build.VERSION.SDK_INT >= Build.VERSION_CODES.KITKAT){
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
            }
        }
    }

    /**
     * 在{@link #previewView}第一次布局完成后，去除该监听，并且进行引擎和相机的初始化
     */
    @Override
    public void onGlobalLayout() {

    }

    @Override
    public void onPreviewFrame(byte[] bytes, Camera camera) {
        if(mDetectState != DETECT_STATE_FREE ){
            return;
        }

        if(mVideoCurrent + 500 >= mVideoMillSecondsLength){
            return;
        }

        mVideoCurrent += 500;
        mMediaDecoder.decodeFrame(mVideoCurrent, mOnGetBitmapImpl);
    }

    private void loadDetectVideo(){
        String fileName = "Rec0003.mp4";
        Log.e("LWS", "load video..."+fileName);

        String filePath = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_MOVIES) + "/" + fileName;
        mMediaDecoder = new MediaDecoder(filePath);

        mVideoMillSecondsLength = Integer.parseInt(mMediaDecoder.getVedioFileLength());
        mVideoCurrent = 0;
    }

    private void showPicture(String imageName){

        try {
            Log.e("LWS", "load image...");
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
        Log.e("LWS", "load image...done");

        // 进行预测
        Log.e("LWS", "predict...");
        ArrayList<Classifier.Recognition> recognitions = predict(0, mBitmap);
        Log.e("LWS", "predict...done");

        // 绘制识别结果
        Log.e("LWS", "draw result...");
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
        Log.e("LWS", "draw result...done");
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
            Log.e("= Detect result(2)", ">>>>>>>>>>>>>>>>>>>> \n" + result);

            return recognitions;
        }

        return null;
    }
}
