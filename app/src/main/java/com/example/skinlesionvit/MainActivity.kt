package com.example.skinlesionvit

import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream
import kotlin.concurrent.thread
import android.util.Log


class MainActivity : AppCompatActivity() {

    private var model: Module? = null
    private lateinit var imagePreview: ImageView
    private lateinit var resultText: TextView
    private lateinit var uploadButton: Button

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
        uploadButton = findViewById(R.id.uploadButton)

        uploadButton.isEnabled = false
        resultText.text = "Loading model..."

        thread {
            try {
                val file = File(filesDir, "lesion_model.ptl")  // ← FIXED
                if (!file.exists()) {
                    assets.open("lesion_model.ptl").use { input ->  // ← FIXED
                        FileOutputStream(file).use { output -> input.copyTo(output) }
                    }
                }
                model = LiteModuleLoader.load(file.absolutePath)

                runOnUiThread {
                    uploadButton.isEnabled = true
                    resultText.text = "Ready"
                    Toast.makeText(this, "Model loaded", Toast.LENGTH_SHORT).show()
                }
            } catch (e: Exception) {
                runOnUiThread {
                    resultText.text = "ERROR: ${e.message}"
                }
            }
        }

        uploadButton.setOnClickListener {
            val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
            pickImage.launch(intent)
        }
    }


    //used softmax
    private fun predictLesion(bitmap: Bitmap) {
        val m = model ?: run {
            resultText.text = "Model not loaded"
            return
        }

        resultText.text = "Analyzing..."
        thread {
            try {
                // Resize to match ViT input
                val resized = Bitmap.createScaledBitmap(bitmap, 256, 256, true)

                // Convert to tensor with normalization
                val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
                    resized,
                    floatArrayOf(0.485f, 0.456f, 0.406f),
                    floatArrayOf(0.229f, 0.224f, 0.225f)
                )

                // Run inference
                val output = m.forward(IValue.from(inputTensor)).toTensor()
                val scores = output.dataAsFloatArray
//                Log.d("MODEL_DEBUG", "Output size: ${scores.size}")
                Log.d("ModelOutput", "Raw logits: ${scores.joinToString()}")



                // Softmax to get probabilities
                val expScores = scores.map { Math.exp(it.toDouble()) }
                val sumExp = expScores.sum()
                val probs = expScores.map { (it / sumExp).toFloat() }

                // 5️⃣ Define your 14 class names (from your dataset)
                val classNames = arrayOf(
                    "Actinic keratoses",
                    "Basal cell carcinoma",
                    "Benign keratosis-like lesions",
                    "Chickenpox",
                    "Cowpox",
                    "Dermatofibroma",
                    "HFMD",
                    "Healthy",
                    "Measles",
                    "Melanocytic nevi",
                    "Melanoma",
                    "Monkeypox",
                    "Squamous cell carcinoma",
                    "Vascular lesions"
                )

                // 6️⃣ Find the top prediction
                val bestIndex = probs.indices.maxByOrNull { probs[it] } ?: 0
                val bestLabel = classNames[bestIndex]
                val bestProb = probs[bestIndex] * 100


                // 7️⃣ Display result on UI thread
                runOnUiThread {
                    resultText.text = "$bestLabel (prob: ${"%.2f".format(bestProb)}%)"
//                    resultText.text = "Raw logits:\n${scores.joinToString(", ")}"
                }


            } catch (e: Exception) {
                runOnUiThread { resultText.text = "Error: ${e.message}" }
                e.printStackTrace()
            }
        }
    }






}