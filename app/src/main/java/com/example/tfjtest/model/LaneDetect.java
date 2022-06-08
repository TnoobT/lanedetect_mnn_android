package com.example.tfjtest.model; // tfj,接口，调用ini_interface.cpp

import android.graphics.Bitmap;

public class LaneDetect {

    static {
        System.loadLibrary("tengmnn"); // tfj,调用生成的库
    }

    public static native void init(String name, String path, boolean useGPU); // tfj,调用 jni_interface.cpp
    public static native LaneInfo[] detect(Bitmap bitmap, byte[] imageBytes, int width, int height, double threshold, double lens_threshold);
}
