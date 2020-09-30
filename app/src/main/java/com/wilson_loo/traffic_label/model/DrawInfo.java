package com.arcsoft.trafficLabel.model;

import android.graphics.Bitmap;
import android.graphics.Rect;
import android.os.Bundle;

public class DrawInfo {
    private Rect rect;
    private int sex;
    private int age;
    private int liveness;
    private int color;
    private String name = null;
    private Bundle bundle = null;

    public DrawInfo(Rect rect, int sex, int age,int liveness,int color,String name, Bundle bundle) {
        this.rect = rect;
        this.sex = sex;
        this.age = age;
        this.liveness = liveness;
        this.color = color;
        this.name = name;
        this.bundle = bundle;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public Rect getRect() {
        return rect;
    }

    public void setRect(Rect rect) {
        this.rect = rect;
    }

    public int getSex() {
        return sex;
    }

    public void setSex(int sex) {
        this.sex = sex;
    }

    public int getAge() {
        return age;
    }

    public void setAge(int age) {
        this.age = age;
    }

    public int getLiveness() {
        return liveness;
    }

    public void setLiveness(int liveness) {
        this.liveness = liveness;
    }

    public int getColor() {
        return color;
    }

    public void setColor(int color) {
        this.color = color;
    }

    public Bundle getBundle() { return this.bundle;}
    public void setBundle(Bundle bundle){this.bundle = bundle;}
}
