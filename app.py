from flask import Flask, render_template, request
from werkzeug.utils import secure_filename 
import os
from pathlib import Path
import torch
import torchvision 
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as patches
from torchvision.ops import nms
import os


def create_detection_model(num_classes:int=2):
  """creates a FRCNN-resnet object detection model"""
  weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
  transforms= weights.transforms()
  model= torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights)
  in_features=model.roi_heads.box_predictor.cls_score.in_features
  for params in model.parameters():
    params.requires_grad=False
  model.roi_heads.box_predictor=FastRCNNPredictor(in_features,num_classes)
  return model,transforms

#function to create efficientnet model for classificatiion..
def create_classification_model():
  weights=torchvision.models.ViT_B_16_Weights.DEFAULT
  transforms=weights.transforms()
  model=torchvision.models.vit_b_16(weights=weights)
  for params in model.parameters():
    params.requires_grad=False
  model.heads.head=torch.nn.Linear(in_features=768,out_features=6,bias=True)
  return model,transforms


#creating the models amd then loading the weights and biases 
det_model,det_transforms=create_detection_model()
det_model.load_state_dict(torch.load("E:\Downloads\Faster_rcnn_model(5).pth",map_location=torch.device("cpu")))

cls_model,cls_transforms=create_classification_model()
cls_model.load_state_dict(torch.load("D:\classification_vitb16_model(1).pth",map_location=torch.device("cpu")))


#returns tensors of the cropped region 
def crop_and_resize_image(image, bbox, output_size=(224, 224)):
  """
  Crop and resize the ROI from the image using the bounding box coordinates.

  Args:
  - image (PIL.Image): Input image.
  - bbox (list or tuple): Bounding box coordinates in the format (x_min, y_min, x_max, y_max).
  - output_size (tuple): Desired output size in the format (height, width).

  Returns:
  - cropped_image_tensor (torch.Tensor): Tensor containing the resized cropped ROI.
  """
  x_min, y_min, x_max, y_max = bbox

  # Crop ROI from the image
  cropped_image = image.crop((x_min, y_min, x_max, y_max))

  # Resize cropped ROI
  transform = T.Compose([
      T.Resize(output_size),
      T.ToTensor(),
  ])

  cropped_image_tensor = transform(cropped_image)

  return cropped_image_tensor


"""def predict_and_plot_bounding_boxes(model, image_path):
  # Load and transform the image
  image = Image.open(image_path).convert("RGB")
  transform = T.Compose([T.ToTensor()])
  image_tensor = transform(image).unsqueeze(0) # Add batch dimension and move to device

  # Make prediction
  model.eval()
  with torch.no_grad():
    predictions = model(image_tensor)

  # Convert predictions to list of dictionaries
  predicted_boxes = [{
      "boxes": pred["boxes"].cpu().numpy(),
      "labels": pred["labels"].cpu().numpy(),
      "scores": pred["scores"].cpu().numpy(),
  } for pred in predictions]

  box_coordinates = predicted_boxes[0]["boxes"]  # Assuming a single image prediction
  scores = predicted_boxes[0]["scores"]

  # Perform Non-Maximum Suppression
  keep = nms(torch.tensor(box_coordinates), torch.tensor(scores), iou_threshold=0.05)
  filtered_boxes = [box for idx, box in enumerate(box_coordinates) if idx in keep]

  cropped_images = []
  for box in filtered_boxes:
    cropped_image_tensor = crop_and_resize_image(image, box)
    cropped_images.append(cropped_image_tensor)
  # Plot the image and filtered bounding boxes
  fig, ax = plt.subplots(1)
  ax.imshow(image)

  for box in filtered_boxes:
    x1, y1, x2, y2 = box
    print(f"Scaled Box coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")  # Debugging print

    # Create a Rectangle patch
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='None')

    # Add the patch to the Axes
    ax.add_patch(rect)
  ax.axis("off")
  plt.show()

#this is the function call for detection of fruits"""


def predict_and_plot_bounding_boxes(model, image_path):
  plt.ioff()
  original_filename = os.path.basename(image_path)
  # Load and transform the image
  image = Image.open(image_path).convert("RGB")
  transform = T.Compose([T.ToTensor()])
  image_tensor = transform(image).unsqueeze(0) # Add batch dimension and move to device

  # Make prediction
  model.eval()
  with torch.inference_mode():
    predictions = model(image_tensor)

  # Convert predictions to list of dictionaries
  predicted_boxes = [{
      "boxes": pred["boxes"].cpu().numpy(),
      "labels": pred["labels"].cpu().numpy(),
      "scores": pred["scores"].cpu().numpy(),
  } for pred in predictions]

  box_coordinates = predicted_boxes[0]["boxes"]  # Assuming a single image prediction
  scores = predicted_boxes[0]["scores"]

  # Perform Non-Maximum Suppression
  keep = nms(torch.tensor(box_coordinates), torch.tensor(scores), iou_threshold=0.13)
  filtered_boxes = [box for idx, box in enumerate(box_coordinates) if idx in keep]

  cropped_images = []
  for box in filtered_boxes:
    cropped_image_tensor = crop_and_resize_image(image, box)
    cropped_images.append(cropped_image_tensor)
  # Plot the image and filtered bounding boxes
  fig, ax = plt.subplots(1)
  ax.imshow(image)

  for box in filtered_boxes:
    x1, y1, x2, y2 = box
    print(f"Scaled Box coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")  # Debugging print

    # Create a Rectangle patch
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='None')

    # Add the patch to the Axes
    ax.add_patch(rect)
  ax.axis("off")
  #plt.show()
  # Save the plot as an image
  save_dir = "static\plots"  # Directory to save the plots
  os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
  save_path = os.path.join(save_dir, original_filename.replace(".jpg", "_bounding_boxes.png"))  # Replace .jpg with the actual image extension
  plt.savefig(save_path, format='png')
  
  plt.close(fig)  # Close the plot to free memory
  
  return save_path
def classify_image(image,model,classes):

  image=image.unsqueeze(dim=0)
  model.eval()
  with torch.inference_mode():
    pred=model(image)#we get the logits here
    pred_probs=torch.softmax(pred,dim=1)
    class_label=torch.argmax(pred_probs,dim=1).item()
  return classes[class_label]

def predict_bounding_boxes(model, image_path):
  # Load and transform the image
  image = Image.open(image_path).convert("RGB")
  transform = T.Compose([T.ToTensor()])
  image_tensor = transform(image).unsqueeze(0) # Add batch dimension and move to device

  # Make prediction
  model.eval()
  with torch.inference_mode():
    predictions = model(image_tensor)

  # Convert predictions to list of dictionaries
  predicted_boxes = [{
      "boxes": pred["boxes"].cpu().numpy(),
      "labels": pred["labels"].cpu().numpy(),
      "scores": pred["scores"].cpu().numpy(),
  } for pred in predictions]

  box_coordinates = predicted_boxes[0]["boxes"]  # Assuming a single image prediction
  scores = predicted_boxes[0]["scores"]

  # Perform Non-Maximum Suppression
  keep = nms(torch.tensor(box_coordinates), torch.tensor(scores), iou_threshold=0.13)
  filtered_boxes = [box for idx, box in enumerate(box_coordinates) if idx in keep]
  filtered_scores=[score for idx,score in enumerate(scores)  if idx in keep]

  cropped_images = []
  for box in filtered_boxes:
    cropped_image_tensor = crop_and_resize_image(image, box)
    cropped_images.append(cropped_image_tensor)
  return cropped_images,filtered_boxes,filtered_scores




def predict(image_dir):
  classes=["it is infloroscence requires long time.","May require around 2.5 Months to ripen ","May require somewhere around 1.5 Months to ripen","May require around 1 Month to ripen","May require around 20 Days to ripen","Ready to harvest!!"]
  pred_classes=[]
  results={"id":[],"time_required":[]}
  cropped_tensors,boxes,scores=predict_bounding_boxes(det_model,image_dir)
  for img_tensor in cropped_tensors:
    pred_class=classify_image(img_tensor,cls_model,classes)
    pred_classes.append(pred_class)
  for id,classes in enumerate(pred_classes):
     results["id"].append(id+1)
     results["time_required"].append(classes) 

  save_file_path=plot_bboxes(image_dir,boxes,pred_classes,scores)
  return results,save_file_path





def plot_bboxes(image_dir, boxes, classes, scores=None, figsize=(10, 10)):
    """
    Plot bounding boxes on the image with class names.

    Args:
    - image_dir (str): Path to the input image.
    - boxes (list): List of bounding box coordinates [(x1, y1, x2, y2), ...].
    - classes (list): List of length N containing the class names corresponding to the bounding boxes.
    - scores (list, optional): List of confidence scores for each bounding box.
    - figsize (tuple, optional): Figure size (width, height).

    Returns:
    - None: Displays the plot.
    """
    original_filename = os.path.basename(image_dir)
    T = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    image = Image.open(image_dir)
    image = T(image)

    # Convert tensor to numpy array
    image = image.numpy()

    # Create figure and axes
    fig, ax = plt.subplots(1, figsize=figsize)

    # Display the image
    ax.imshow(np.transpose(image, (1, 2, 0)))

    # Iterate over all bounding boxes
    for i, box in enumerate(boxes):
        class_name = classes[i]

        # Extract coordinates
        x1, y1, x2, y2 = box

        # Calculate box width and height
        width = x2 - x1
        height = y2 - y1

        # Create a rectangle patch
        rect = patches.Rectangle((x1, y1), width, height, linewidth=1, edgecolor='r', facecolor='none')

        # Add the patch to the plot
        ax.add_patch(rect)

        # Add class name and score (if available) to the plot
        if scores is not None:
            score = scores[i]
            ax.text(x1, y1 - 5, f"id:{i+1}", color='blue', fontsize=20)
        else:
            ax.text(x1, y1 - 5, f"id:{i+1}", color='blue', fontsize=20)

    # Set axis properties
    ax.axis('off')

    # Show plot
    #plt.show()
    save_dir = "static/results"  # Directory to save the plots
    os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
    save_path = os.path.join(save_dir, original_filename.replace(".jpg", "_class_id__boxes.jpg").replace("\\", "/"))  # Replace .jpg with the actual image extension
    plt.savefig(save_path, format='jpg')
    
    plt.close(fig)  # Close the plot to free memory
  
    return save_path
#this is the function call to classify the detected fruits


app=Flask(__name__)




upload_folder= os.path.join("static","uploads")

app.config["UPLOAD"]= upload_folder

@app.route("/",methods=["GET", "POST"])
def index():
    return render_template("image_render.html")

@app.route("/upload",methods=["POST"])
def upload_file():
    if request.method=="POST":
        file=request.files["img"]
        filename= secure_filename(file.filename)
        file.save(os.path.join(app.config["UPLOAD"],filename))
        img= os.path.join(app.config["UPLOAD"],filename)
        return render_template("image_render.html",img=img)
    
  
@app.route("/process",methods=["POST"])
def process():
    if request.method=="POST":
        img_file=request.form.get("img_path")
        file=predict_and_plot_bounding_boxes(det_model, img_file)
        return render_template("image_render.html",Pimg=file,Aimg=img_file)
    
@app.route("/estimate",methods=["POST"])
def estimate():
    if request.method=="POST":
        img_file=request.form.get("img_path_2")
        result_dict,file=predict(img_file)
        return render_template("image_render.html",image=file,resultant=result_dict)

if __name__=="__main__":
    app.run(debug=True,port=8001)