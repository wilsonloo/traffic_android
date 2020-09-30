package com.arcsoft.trafficLabel.util;


import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.util.Log;

import com.tencent.bugly.crashreport.CrashReport;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

/**
 * Bitmap 帮助类之一
 */
public class BitmapUtils {

    public static Bitmap adjustToSize(Bitmap bitmap, int newWidth, int newHeight){
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();

        // 计算缩放比例
        float scaleWidth = ((float) newWidth) / width;
        float scaleHeight = ((float) newHeight) / height;

        // 取得想要缩放的matrix参数
        Matrix matrix = new Matrix();
        matrix.postScale(scaleWidth, scaleHeight);

        // 得到新的图片
        return Bitmap.createBitmap(bitmap, 0, 0, width, height, matrix, true);
    }

    /**
     * Save Bitmap
     *
     * @param name file name
     * @param bm picture to save
     */
    public static String SaveBitmap(String name, Bitmap bm, Context mContext) {
        //指定我们想要存储文件的地址
        String TargetPath = mContext.getFilesDir() + "/images/";

        //判断指定文件夹的路径是否存在
        if (!FileUtils.fileIsExist(TargetPath)) {
            Log.d("Save Bitmap", "TargetPath isn't exist");
            return null;
        } else {
            //如果指定文件夹创建成功，那么我们则需要进行图片存储操作
            String filePath = TargetPath + name + ".jpg";
            File saveFile = new File(filePath);

            try {
                FileOutputStream saveImgOut = new FileOutputStream(saveFile);

                // compress - 压缩的意思
                bm.compress(Bitmap.CompressFormat.JPEG, 80, saveImgOut);

                //存储完成后需要清除相关的进程
                saveImgOut.flush();
                saveImgOut.close();

                Log.d("Save Bitmap", "write to "+filePath);
                return filePath;

            } catch (IOException ex) {
                CrashReport.postCatchedException(ex);
                ex.printStackTrace();
                Log.d("Save Bitmap", "with exception");
            }
        }

        return null;
    }
}


