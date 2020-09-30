package com.arcsoft.trafficLabel.common;

import com.arcsoft.arcfacedemo.R;

import java.util.HashMap;
import java.util.Map;

public class EmotionResourceMap {
    private final Map<String, Integer> mEmotionResourceMap = new HashMap<>();

    private static EmotionResourceMap mInstance = new EmotionResourceMap();

    private EmotionResourceMap(){
        mEmotionResourceMap.put("anger", R.mipmap.anger);
        mEmotionResourceMap.put("disgust", R.mipmap.disgust);
        mEmotionResourceMap.put("fear", R.mipmap.fear);
        mEmotionResourceMap.put("happy", R.mipmap.happy);
        mEmotionResourceMap.put("sad", R.mipmap.sad);
    }

    public static  EmotionResourceMap getInstance(){
        return mInstance;
    }

    public static Integer getEmotionResource(String emotionType){
        return getInstance().mEmotionResourceMap.get(emotionType);
    }
}
