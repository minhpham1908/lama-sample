package com.tapuniverse.lama_dilated_sample

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.providers.NNAPIFlags
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.Paint
import android.util.Log
import androidx.core.graphics.scale
import java.nio.FloatBuffer
import java.util.EnumSet

class Lama(private val context: Context) {
    private val TAG = "RemoveObjects_Lama"
    private var session: OrtSession? = null

    init {
        initialize()
    }

    private fun initialize() {

        var preStart = System.currentTimeMillis()

        val env = OrtEnvironment.getEnvironment()
        val model = context.assets.open("lama_fp32.with_runtime_opt.ort")
        val modelBytes = model.readBytes()
        val sessionOption =  OrtSession.SessionOptions()
        sessionOption.setInterOpNumThreads(4)
        sessionOption.setIntraOpNumThreads(4)
        session = env.createSession(modelBytes,sessionOption)
        var preEnd = System.currentTimeMillis()
        Log.d(TAG, "initialize: ${preEnd - preStart}")
        Log.d(TAG, "initialize")
    }


    fun run(image: Bitmap, mask: Bitmap): Bitmap? {
        Log.d(TAG, "run: ")
        if (session == null) {
            initialize()
        }
        var preStart = System.currentTimeMillis()
        val imgData = preProcessImage(image.scale(512, 512))
        val maskData = preProcessMask(mask.scale(512, 512))
        val imageShape = longArrayOf(1, 3, 512, 512)
        val maskShape = longArrayOf(1, 1, 512, 512)
        var preEnd = System.currentTimeMillis()

        val env = OrtEnvironment.getEnvironment()
        val imageTensor: OnnxTensor = OnnxTensor.createTensor(env, imgData, imageShape)
        val maskTensor: OnnxTensor = OnnxTensor.createTensor(env, maskData, maskShape)
        val start = System.currentTimeMillis()

        val output = session?.run(mapOf("image" to imageTensor, "mask" to maskTensor), OrtSession.RunOptions())
        val end = System.currentTimeMillis()
        if (output == null || output.size() < 1) {
            return null
        }
        var postStart = System.currentTimeMillis()
        val outputBitmap = postProcessOutput(output)
        var postEnd = System.currentTimeMillis()

        Log.d(TAG, "Pre process run time: ${preEnd - preStart}ms")
        Log.d(TAG, "Session run time: ${end - start}ms")
        Log.d(TAG, "Post process run time: ${postEnd - postStart}ms")
        return outputBitmap
    }

    private fun preProcessImage(bitmap: Bitmap): FloatBuffer {
        val imgData = FloatBuffer.allocate(1 * 3 * 512 * 512)
        imgData.rewind()
        val stride = 512 * 512
        val bmpData = IntArray(stride)
        bitmap.getPixels(bmpData, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        for (i in 0..<512) {
            for (j in 0..<512) {
                val idx = 512 * i + j
                val pixelValue = bmpData[idx]
                imgData.put(idx, (((pixelValue shr 16 and 0xFF) / 255f) / 1f))
                imgData.put(idx + stride, (((pixelValue shr 8 and 0xFF) / 255f - 0) / 1f))
                imgData.put(idx + stride * 2, (((pixelValue and 0xFF) / 255f - 0f) / 1f))
            }
        }

        imgData.rewind()
        return imgData
    }

    private fun preProcessMask(mask: Bitmap): FloatBuffer {
        val h: Int = mask.getHeight()
        val w: Int = mask.getWidth()
        val bmpGrayscale = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bmpGrayscale)
        val paint = Paint()

        val colorMatrix = ColorMatrix()
        colorMatrix.setSaturation(0f)
        val colorMatrixFilter = ColorMatrixColorFilter(colorMatrix)
        paint.setColorFilter(colorMatrixFilter)
        canvas.drawBitmap(mask, 0.0f, 0.0f, paint)

        val imgData = FloatBuffer.allocate(1 * 1 * 512 * 512)
        imgData.rewind()
        val stride = 512 * 512
        val bmpData = IntArray(stride)
        bmpGrayscale.getPixels(bmpData, 0, mask.width, 0, 0, mask.width, mask.height)
        for (i in 0..<512) {
            for (j in 0..<512) {
                val idx = 512 * i + j
                val pixelValue = bmpData[idx]
                val value = (pixelValue shr 16 and 0xFF).toFloat() / 255f
                imgData.put(idx, if (value > 0f) 1f else 0f)
            }
        }
        imgData.rewind()
        return imgData
    }

    private fun postProcessOutput(output: OrtSession.Result): Bitmap {
        val floatArray = (output.get(0) as OnnxTensor).floatBuffer
        val colors = IntArray(512 * 512)
        val stride = 512 * 512
        for (i in 0..<512) {
            for (j in 0..<512) {
                val idx = 512 * j + i
                val red = floatArray[idx].toInt()
                val green = floatArray[idx + stride].toInt()
                val blue = floatArray[idx + stride * 2].toInt()
                colors[idx] = Color.argb(255, red, green, blue)
            }
        }
        val bitmap = Bitmap.createBitmap(colors, 512, 512, Bitmap.Config.ARGB_8888)
        return bitmap
    }

}