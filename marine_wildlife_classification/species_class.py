import torch
from torchvision import models, transforms
from PIL import Image
import os

# --- Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# List of classes (must match training)
class_names = ['malabar_group', 'napoleon_wrass', 'whale_shark']
num_classes = len(class_names)

# --- Load Trained Model ---
model = models.mobilenet_v2(pretrained=False)
num_features = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(num_features, num_classes)

model.load_state_dict(torch.load("marine_species_model.pth", map_location=device))
model = model.to(device)
model.eval()

# --- Prediction Function ---
def predict_species(image_path):
    if not os.path.isfile(image_path):
        print(f"ERROR: Image file not found → {image_path}")
        return None

    image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    image = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return class_names[predicted.item()]

# --- Example Usage ---
if __name__ == "__main__":
    test_images_dir = "test_images"
    
    for filename in os.listdir(test_images_dir):
        image_path = os.path.join(test_images_dir, filename)
        predicted_species = predict_species(image_path)

        if predicted_species is not None:
            print(f"Image: {filename} → Predicted Species: {predicted_species}")

