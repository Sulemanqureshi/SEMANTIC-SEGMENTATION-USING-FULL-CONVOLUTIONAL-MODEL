# SEMANTIC-SEGMENTATION-USING-FULL-CONVOLUTIONAL-MODEL
Imports:
torch: Imports PyTorch, a deep learning library.
torchvision: Imports TorchVision, a package for vision-related tasks in PyTorch.
cv2: Imports OpenCV for image processing.
numpy: Imports NumPy for numerical computations (e.g., handling arrays).
matplotlib.pyplot: Imports Matplotlib for displaying images.
PIL.Image: Imports the Python Imaging Library (PIL) for handling image operations.
Load FCN Model:
Load pre-trained FCN model: Loads the pre-trained FCN model (fcn_resnet101) using the torchvision.models.segmentation module. This model is designed for semantic segmentation.
Set model to evaluation mode: The model is switched to evaluation mode using model.eval(), which disables training-specific features like dropout.
Preprocessing Function (preprocess):
Load the image: Uses OpenCV's cv2.imread() to read the input image from the specified path.
Convert BGR to RGB: Converts the image from OpenCV's BGR format to RGB using cv2.cvtColor().
Resize the image: Resizes the image to (512, 512) to meet the input size requirement of the FCN model using cv2.resize().
Create preprocessing transformations: Uses torchvision.transforms.Compose() to define a transformation pipeline that converts the image to a tensor and normalizes it using ImageNet's mean and standard deviation values.
Convert to tensor: Applies the transformation pipeline using preprocess_transform(), converting the resized image into a PyTorch tensor. Then, the batch dimension is added with .unsqueeze(0).
Return the tensor and resized image: The function returns both the input tensor (for the model) and the resized image (for visualization).
Segmentation Function (segment_image):
Disable gradient computation: Uses torch.no_grad() to disable gradient calculation for faster inference.
Run the model: Passes the input tensor through the FCN model, and extracts the output prediction from the ['out'] key.
Get predicted classes: Takes the argmax of the model’s output over the class dimension to get the predicted class for each pixel. Converts the result to a NumPy array using .cpu().numpy().
Return the segmentation mask: The function returns the segmentation mask, where each pixel contains the class index predicted by the model.
Overlay Mask Function (overlay_segmentation_mask):
Create a random colormap: Creates a random colormap for the 21 classes (the number of object categories in the FCN model).
Apply the colormap: Uses the class index mask to color the corresponding pixels using the colormap.
Blend the original image and mask: Combines the original image and the colored mask using cv2.addWeighted() with a 60-40 blending ratio.
Return the blended image: The function returns the image with the segmentation mask overlay.
Image Processing Pipeline:
Define image path: Specifies the path to the input image.
Preprocess the image: Calls the preprocess() function to obtain the input tensor and resized image.
Segment the image: Calls segment_image() to get the predicted segmentation mask.
Overlay the segmentation mask: Calls overlay_segmentation_mask() to overlay the segmentation mask on the original image.
Display Results:
Set up a plot: Creates a Matplotlib figure to display both the original and segmented images.
Display original image: Displays the original image in the first subplot.
Display segmented image: Displays the image with the segmentation mask in the second subplot.
Show the plot: Calls plt.show() to display the side-by-side comparison of the original and segmented images.
