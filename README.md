# WSRGAN
A Wasserstein Super Resolution Generative Adversarial Network I made by following some tutorials and combining some notes I found online.

The code for the model is a little scattered, as I made it to run from the terminal on the computer I was using to train it. Some versions of the model took up to a week to train. I would refactor the code, but I considered this to be a project for practice more than practical implementation. My model does have better results in terms of details than many others I found online at the time I made it (early to mid 2023), but it was a tad noisy and thus didn't produce very smooth images. The non-gan version that simply used a loss comparison algorithm produced smooth, but decent results. That made that version good for cartoony images, but poorer for photos or complex 3D renders.

I also made a version of the model that only runs on layer of an image at a time (meaning it only upscales the Red, Green, Blue, or Alpha channel) then it combines them all using a script back into a whole image. This is advantageous compared to the non-one layer one, as it can run on images of different layer amounts, as ModelNew31 can only properly run on images with 3 layers, so it only produces images of RGB.

My use of the script:
1. import the python file ("import ModelNew31")
2. resond with 1 to tell it to use gpu
3. tell it to train on images in a directory (Ex. "ModelNew31_OneLayer.train_DeepLearnOnDirectoryRepeat("dataset/train2014/", ModelNew31.model, ModelNew31.modelD, ModelNew31.optimizerG, ModelNew31.optimizerD, True, 1, 16, True, True, True, True, False)")
4. Alternatively I could load a model (Ex. "ModelNew31.loadModel("save_modelNew31.pth")")
5. Could also upscale an image (Ex. "ModelNew31_OneLayer.makeUpscale("testImage.jpg", cap=True)")

The model was made on Python Version 3.10

Some of the references used:
- https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution
  - This one was referenced for the newest version of the model (ModelNew31)
- https://www.youtube.com/watch?v=pG0QZ7OddX4
  - This one was used to implement the wasserstein part of the model
- https://learn.microsoft.com/en-us/training/modules/intro-computer-vision-pytorch/
  - This was used to learn about convolutionary networks or layers

Current Problems In Model:
- Model doesn't seem to be fully converged for the GAN version, as it produces slightly noisy results, but they are detailed
- The model consumes an excessive amount of ram when upscaling images. Possible the issue is from pythons memory management, as it seems to increase memory without freeing earlier memory when scaling larger images
- The model will run out of memory or use an impossible amount if running with parallel cores on cpu as a result of how python manages memory for cpu parallelism (something to do with copy-on-write probably)
