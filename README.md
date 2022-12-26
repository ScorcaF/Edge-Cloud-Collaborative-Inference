# Edge-Cloud-Collaborative-Inference
Collaborative Inference framework with the following components: 
  1. a **Fast inference** pipeline running on edge;
  2. a **Slow inference** pipeline running on the cloud;
  3. a **“success checker” policy** to determine whether the Fast inference was “confident” about its prediction or not; if not, run the Slow inference to get the final     prediction.



![edge cloud](https://user-images.githubusercontent.com/70110839/209355345-1b9b7669-db17-49a7-9c95-b90af3489ea9.png)

Both pipelines run the same model (same architecture and same weights) deployed in tflite format, but with
different pre-processing.
