package com.wilson_loo.traffic_label.activity;

import android.Manifest;
import android.content.Intent;
import android.content.pm.ApplicationInfo;
import android.hardware.Camera;
import android.os.Bundle;
import android.support.v4.app.ActivityCompat;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.RadioGroup;
import android.widget.SeekBar;
import android.widget.Spinner;

import com.arcsoft.trafficLabel.common.Constants;
import com.tencent.bugly.crashreport.CrashReport;
import com.wilson_loo.traffic_label.R;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import io.reactivex.Observable;
import io.reactivex.ObservableEmitter;
import io.reactivex.ObservableOnSubscribe;
import io.reactivex.Observer;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.disposables.Disposable;
import io.reactivex.schedulers.Schedulers;

public class ChooseFunctionActivity extends BaseActivity {
    private static final int ASF_OP_0_ONLY = 0;
    private static final int ASF_OP_90_ONLY = 1;
    private static final int ASF_OP_180_ONLY = 2;
    private static final int ASF_OP_270_ONLY = 3;
    private static final int ASF_OP_ALL_OUT = 4;

    private static final String TAG = "ChooseFunctionActivity";
    private static final int ACTION_REQUEST_PERMISSIONS = 0x001;
    // 在线激活所需的权限
    private static final String[] NEEDED_PERMISSIONS = new String[]{
            Manifest.permission.READ_PHONE_STATE
    };
    boolean libraryExists = true;
    // Demo 所需的动态库文件
    private static final String[] LIBRARIES = new String[]{
            // 人脸相关
            "libarcsoft_face_engine.so",
            "libarcsoft_face.so",
            // 图像库相关
            "libarcsoft_image_util.so",
    };

    // 摄像头索引
    private int mCameraIndex = Camera.CameraInfo.CAMERA_FACING_BACK;
    private int mTensorflowType = Constants.TENSORFLOW_TYPE_TFLITE;
    private SeekBar mSeekBar = null;

    private RadioGroup.OnCheckedChangeListener mRadioGroupChooseCameraListener = new RadioGroup.OnCheckedChangeListener(){
        @Override
        public void onCheckedChanged(RadioGroup group, int checkedId) {

            int id = group.getCheckedRadioButtonId();
            switch (group.getCheckedRadioButtonId()) {
                case R.id.rb_back_camera:
                    mCameraIndex = Camera.CameraInfo.CAMERA_FACING_BACK;
                    break;
                case R.id.rb_front_camera:
                    mCameraIndex = Camera.CameraInfo.CAMERA_FACING_FRONT;
                    break;
                default:
                    assert false;
                    break;
            }
        }
    };

    private Spinner.OnItemSelectedListener mSpinnerTFliteListener = new Spinner.OnItemSelectedListener(){

        @Override
        public void onItemSelected(AdapterView<?> adapterView, View view, int tensorflowType, long l) {
            mTensorflowType = tensorflowType;
        }

        @Override
        public void onNothingSelected(AdapterView<?> adapterView) {

        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_choose_function);
        libraryExists = checkSoFile(LIBRARIES);
        ApplicationInfo applicationInfo = getApplicationInfo();
        Log.i(TAG, "onCreate: " + applicationInfo.nativeLibraryDir);
        if (!libraryExists) {
            showToast(getString(R.string.library_not_found));
        } else {
            initView();
        }

        // 表情检测的摄像头选择
        RadioGroup radioGroupChooseCamera =  findViewById(R.id.radio_group_choose_camera);
        radioGroupChooseCamera.check(R.id.rb_back_camera);
        radioGroupChooseCamera.setOnCheckedChangeListener(mRadioGroupChooseCameraListener);

        // tflite type
        Spinner spinnerTflite = findViewById(R.id.spinner_tensorflows);
        spinnerTflite.setOnItemSelectedListener(mSpinnerTFliteListener);

        // 积分阈值
        mSeekBar= findViewById(R.id.scoreThresoldSeekBar);
        mSeekBar.setMax(100);
        mSeekBar.setProgress(80);
        mSeekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int i, boolean b) {

            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {

            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {

            }
        });

        CrashReport.initCrashReport(getApplicationContext(), "65b12e9090", true);

//        testStaticEmotionClassify();
    }

    /**
     * 检查能否找到动态链接库，如果找不到，请修改工程配置
     *
     * @param libraries 需要的动态链接库
     * @return 动态库是否存在
     */
    private boolean checkSoFile(String[] libraries) {
        File dir = new File(getApplicationInfo().nativeLibraryDir);
        File[] files = dir.listFiles();
        if (files == null || files.length == 0) {
            return false;
        }
        List<String> libraryNameList = new ArrayList<>();
        for (File file : files) {
            libraryNameList.add(file.getName());
        }
        boolean exists = true;
        for (String library : libraries) {
            exists &= libraryNameList.contains(library);
        }
        return exists;
    }

    private void initView() {

    }

    /**
     * 开始实时检测人脸表情
     *
     * @param view
     */
    public void jumpToDetectFaceEmotionActivity(View view) {
        Bundle bundle = new Bundle();
        bundle.putInt("whichCamera", mCameraIndex);
        bundle.putInt("tensorflowType", mTensorflowType);
        bundle.putFloat("scoreThreshold", mSeekBar.getProgress() * 1.0f / 100);
        checkLibraryAndJump(DetectTrafficLabelActivity.class, bundle);
    }

    /**
     * 激活引擎
     *
     * @param view
     */
    public void activeEngine(final View view) {
        if (!libraryExists) {
            showToast(getString(R.string.library_not_found));
            return;
        }
        if (!checkPermissions(NEEDED_PERMISSIONS)) {
            ActivityCompat.requestPermissions(this, NEEDED_PERMISSIONS, ACTION_REQUEST_PERMISSIONS);
            return;
        }
        if (view != null) {
            view.setClickable(false);
        }
        Observable.create(new ObservableOnSubscribe<Integer>() {
                    @Override
                    public void subscribe(ObservableEmitter<Integer> emitter) {

                    }
                })
                .subscribeOn(Schedulers.io())
                .observeOn(AndroidSchedulers.mainThread())
                .subscribe(new Observer<Integer>() {
                    @Override
                    public void onSubscribe(Disposable d) {

                    }

                    @Override
                    public void onNext(Integer activeCode) {

                    }

                    @Override
                    public void onError(Throwable e) {
                        showToast(e.getMessage());
                        if (view != null) {
                            view.setClickable(true);
                        }
                    }

                    @Override
                    public void onComplete() {

                    }
                });
    }

    @Override
    void afterRequestPermission(int requestCode, boolean isAllGranted) {
        if (requestCode == ACTION_REQUEST_PERMISSIONS) {
            if (isAllGranted) {
                activeEngine(null);
            } else {
                showToast(getString(R.string.permission_denied));
            }
        }
    }

    void checkLibraryAndJump(Class activityClass, Bundle bundle) {
        if (!libraryExists) {
            showToast(getString(R.string.library_not_found));
            return;
        }

        Intent intent = new Intent(this, activityClass);
        intent.putExtras(bundle);
        startActivity(intent);
    }
}
