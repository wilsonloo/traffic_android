package com.wilson_loo.traffic_label.util;

import android.graphics.Bitmap;
import android.media.MediaMetadataRetriever;
import android.util.Log;
import com.wilson_loo.traffic_label.util.FileUtils;

public class MediaDecoder {
    private static final String TAG = "MediaDecoder";
    private MediaMetadataRetriever retriever = null;
    private String fileLength;

    public MediaDecoder(String file) {
        if(FileUtils.fileIsExist(file)){
            retriever = new MediaMetadataRetriever();
            retriever.setDataSource(file);
            fileLength = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION);
            Log.i(TAG, "fileLength : "+fileLength);
        }
    }
    /**
     * 获取视频某一帧
     * @param timeMs 毫秒
     * @param listener
     */
    public boolean decodeFrame(long timeMs, OnGetBitmapListener listener){
        if(retriever == null)
            return false;

        Bitmap bitmap = retriever.getFrameAtTime(timeMs * 1000, MediaMetadataRetriever.OPTION_CLOSEST);
        if(bitmap == null)
            return false;

        listener.getBitmap(bitmap, timeMs);
        return true;
    }
    /**
     * 取得视频文件播放长度
     * @return
     */
    public String getVedioFileLength(){
        return fileLength;
    }
}
