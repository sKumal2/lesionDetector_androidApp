package com.example.skinlesionvit

import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import com.example.skinlesionvit.R
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

class MainActivity : AppCompatActivity() {
    private lateinit var model: Module
    private lateinit var imagePreview: ImageView
    private lateinit var resultText: TextView

    private val pickImage = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
        if (result.resultCode == RESULT_OK) {
            val uri: Uri? = result.data?.data
            uri?.let {
                val bitmap = MediaStore.Images.Media.getBitmap(contentResolver, it)
                imagePreview.setImageBitmap(bitmap)
                predictLesion(bitmap)
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        imagePreview = findViewById(R.id.imagePreview)
        resultText = findViewById(R.id.resultText)
        val uploadButton: Button = findViewById(R.id.uploadButton)
        uploadButton.setOnClickListener {
            val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
            pickImage.launch(intent)
        }

        // Load model
        try {
            model = LiteModuleLoader.load(assetFilePath("lesion_model.ptl"))
        } catch (e: IOException) {
            e.printStackTrace()
        }
    }

    private fun predictLesion(bitmap: Bitmap) {
        // Preprocess: Resize to 224x224, normalize (must match Python transforms)
        val inputTensor: Tensor = TensorImageUtils.bitmapToFloat32Tensor(
            bitmap,
            TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,  // [0.485f, 0.456f, 0.406f]
            TensorImageUtils.TORCHVISION_NORM_STD_RGB   // [0.229f, 0.224f, 0.225f]
        )

        // Inference
        val outputTensor = model.forward(IValue.from(inputTensor)).toTensor()
        val scores = outputTensor.dataAsFloatArray  // [no_lesion_prob, lesion_prob]

        val maxScoreIndex = scores.indices.maxByOrNull { scores[it] } ?: 0
        val result = if (maxScoreIndex == 1) "Lesion detected (prob: ${scores[1] * 100}%)" else "No lesion (prob: ${scores[0] * 100}%)"
        resultText.text = result
    }

    @Throws(IOException::class)
    private fun assetFilePath(assetName: String): String {
        val file = File(filesDir, assetName)
        if (file.exists() && file.length() > 0) return file.absolutePath

        assets.open(assetName).use { inputStream ->
            FileOutputStream(file).use { outputStream ->
                val buffer = ByteArray(4 * 1024)
                var read: Int
                while (inputStream.read(buffer).also { read = it } != -1) {
                    outputStream.write(buffer, 0, read)
                }
                outputStream.flush()
            }
        }
        return file.absolutePath
    }
}
