package com.tapuniverse.lama_dilated_sample

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.os.Bundle
import android.util.Log
import android.widget.ImageView
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import org.tensorflow.lite.DataType
import org.tensorflow.lite.InterpreterApi
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.TensorOperator
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.TransformToGrayscaleOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import org.tensorflow.lite.support.tensorbuffer.TensorBufferFloat


class MainActivity : AppCompatActivity() {
    private var interpreter: InterpreterApi? = null
    private val TAG = "MainActivity"
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }
        var start = System.currentTimeMillis()
        var lama = Lama(this)
        var inputImage = BitmapFactory.decodeStream(assets.open("img 1.jpg"))
        var maskImage = BitmapFactory.decodeStream(assets.open("mask.png"))
        var output = lama.run(inputImage  , maskImage)
        findViewById<ImageView>(R.id.imageView).setImageBitmap(output)
//        initModel()
//        onInitialized()
        var endRange = System.currentTimeMillis()

        Log.d(TAG, "Total run time: ${endRange - start}")
    }


    fun initModel() {
        val start = System.currentTimeMillis()
        var modelFile = FileUtil.loadMappedFile(this, "LaMa-Dilated.tflite")
        val interpreterOptions = InterpreterApi.Options()
        interpreterOptions.setNumThreads(4)

//        interpreterOptions.addDelegate(nnApiDelegate)
        interpreter = InterpreterApi.create(modelFile, interpreterOptions)
        val end = System.currentTimeMillis()
        Log.d(TAG, "initModel: ${end - start}")

    }

    private fun onInitialized() {
        val startPre = System.currentTimeMillis()
        var tensorImage = TensorImage(DataType.FLOAT32)
        var tensorMask = TensorImage(DataType.FLOAT32)

        var inputImage = BitmapFactory.decodeStream(assets.open("img.png"))
        var maskImage = BitmapFactory.decodeStream(assets.open("mask.png"))
        tensorImage.load(inputImage)

        var imagePro =
            ImageProcessor.Builder().add(ResizeOp(512, 512, ResizeOp.ResizeMethod.BILINEAR)).add(NormalizeOp(0f, 255f))
                .build()


        tensorImage = imagePro.process(tensorImage)

        var maskProcessor = ImageProcessor.Builder().add(TransformToGrayscaleOp()).add(NormalizeOp(0f, 255f)).add(
            TensorOperator { input ->
                val data = input.floatArray
                for (i in data.indices) {
                    data[i] = if (data[i] > 0f) 1f else 0f
                }
                val output = TensorBufferFloat.createFixedSize(input.shape, DataType.FLOAT32)
                output.loadArray(data, input.shape)
                return@TensorOperator output
            }).build()
        val maskBuffer = TensorBuffer.createFixedSize(intArrayOf(1, 512, 512, 1), DataType.FLOAT32)
        val intValues = IntArray(maskImage.height * maskImage.width)
        val floatValue = FloatArray(maskImage.height * maskImage.width)
        maskImage.getPixels(intValues, 0, maskImage.width, 0, 0, maskImage.width, maskImage.height)
        for (i in 0 until maskImage.height * maskImage.width) {
            val value = (intValues[i] shr 16 and 0xFF)
            floatValue[i] = if (value > 0f) 1f else 0f
            intValues[i] = value

        }
        maskBuffer.loadArray(intValues, intArrayOf(1, 512, 512, 1))

        tensorMask.load(maskImage)
        tensorMask = maskProcessor.process(tensorMask)

        val outputTensor = TensorBuffer.createFixedSize(intArrayOf(1, 512, 512, 3), DataType.FLOAT32)
        val endPre = System.currentTimeMillis()

        var runstart = System.currentTimeMillis()
        interpreter?.runForMultipleInputsOutputs(
            arrayOf(tensorImage.buffer, tensorMask.buffer), mapOf(0 to outputTensor.buffer)
        )
        var runend = System.currentTimeMillis()
        val starPos = System.currentTimeMillis()

        val floatArray = outputTensor.floatArray
        val colors = IntArray(outputTensor.floatArray.size / 3)
        for (i in 0 until outputTensor.floatArray.size step 3) {
            colors[i / 3] = Color.argb(1f, floatArray[i], floatArray[i + 1], floatArray[i + 2])
        }
        val bitmap = Bitmap.createBitmap(colors, 512, 512, Bitmap.Config.ARGB_8888)
        val runPos = System.currentTimeMillis()

        findViewById<ImageView>(R.id.imageView).setImageBitmap(bitmap)
        Log.d(TAG, "Pre time: ${endPre - startPre}")
        Log.d(TAG, "Run time: ${runend - runstart}")
        Log.d(TAG, "Pos time: ${runPos - starPos}")
        Log.d(TAG, " Set bitmap")
    }
}