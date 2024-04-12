import torch
import torchvision.transforms as transforms
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v2 # Placeholder for MobileFormer
import os

# Assuming COCO dataset is stored in 'coco_path'
coco_path = 'path/to/coco'
assert os.path.exists(coco_path), "COCO path does not exist."

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# COCO Detection Dataset (using as a classification dataset for this example)
train_set = CocoDetection(root=os.path.join(coco_path, 'train2017'),
                          annFile=os.path.join(coco_path, 'annotations/instances_train2017.json'),
                          transform=transform)

def collate_fn(batch):
    images, targets = list(zip(*batch))
    images = torch.stack(images, 0)
    labels = torch.tensor([t[0]['category_id'] for t in targets], dtype=torch.long) # Simplified approach
    return images, labels

train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=4, collate_fn=collate_fn)

# Model Placeholder for MobileFormer
# In actual implementation, you would replace mobilenet_v2 with MobileFormer.
model = mobilenet_v2(pretrained=True)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 91) # COCO has 91 classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss and Optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training Loop
model.train()
for epoch in range(1): # For demonstration, run for 1 epoch
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Loss: {loss.item()}')

# Save the trained model
torch.save(model.state_dict(), 'mobile_former_coco.pth')
