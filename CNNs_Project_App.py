import streamlit as st
import joblib as jb
import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import gdown  # Changed to use gdown for downloading files

st.set_page_config(page_icon='🍎', page_title='Fruits And Vegetables Model')
tab1, tab2 = st.tabs(['Introduction', 'Models'])

with tab1:
    st.header('Introduction')
    st.markdown('#### **In this project, 2 different CNN models were trained on approximately 29,000 images.**')
    st.markdown('##### These images included 14 types of vegetables and fruits, divided into 28 categories.')
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    for col, fruit, healthy_img, rotten_img in zip(
        [col1, col2, col3, col4],
        ["Mango", "Apple", "Banana", "Potato"],
        ["photo display/mango_helthy.jpg", "photo display/FreshApple (1).jpg", "photo display/Banana__Healthy_augmented_12.jpg", "photo display/freshPotato (1).jpeg"],
        ["photo display/mango_rotten.jpeg", "photo display/apple_rotten.jpg", "photo display/banana_rotten.png", "photo display/potato_rotten.jpeg"]
    ):
        col.write(fruit)
        sub_col1, sub_col2 = col.columns([1, 1])
        with sub_col1:
            st.write('Healthy')
            st.image(healthy_img)
        with sub_col2:
            st.write('Rotten')
            st.image(rotten_img)
    
    col5, col6, col7 = st.columns([1, 1, 1])
    
    for col, fruit, healthy_img, rotten_img in zip(
        [col5, col6, col7],
        ["Pepper", "Orange", "Tomato"],
        ["photo display/freshPepper (1).jpeg", "photo display/freshOrange (1).jpg", "photo display/freshTomato (5).png"],
        ["photo display/rottenPepper (140).jpg", "photo display/rottenOrange (77).jpg", "photo display/rottenTomato (57).jpg"]
    ):
        col.write(fruit)
        sub_col1, sub_col2 = col.columns([1, 1])
        with sub_col1:
            st.write('Healthy')
            st.image(healthy_img)
        with sub_col2:
            st.write('Rotten')
            st.image(rotten_img)

    st.markdown("""#### 2 different approaches were used in this project:
\n1-TensorFlow
\n2-PyTorch
\n**For the third approach**, the focus was solely on the method. This approach did not yield favorable results, however, there are ways to improve this model. Updates will be made in the near future.""")

with tab2:
    def download_file_from_google_drive(id, destination):
        URL = f"https://drive.google.com/uc?id={id}"
        gdown.download(URL, destination, quiet=False)

    class MyModel(nn.Module):
        def __init__(self):
            super(MyModel, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.fc1 = nn.Linear(64 * 56 * 56, 128)  # Adjust based on your input image size
            self.fc2 = nn.Linear(128, 10)  # Assuming 10 classes

        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = x.view(-1, 64 * 56 * 56)  # Flatten the tensor
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    model_torch = MyModel()
    model_torch.load_state_dict(torch.load('28_EfficientNet_97.pth'), strict=False)

    classes_name = jb.load('classes_name.pkl')

    with st.spinner('Loading TensorFlow model (VGG16)'):
        # File uploader for the TensorFlow model
        model_id = "1hz-vGWfZOQa1EjtKq_6965VNhRg4HqaT"
        download_file_from_google_drive(model_id, '28_VGG16_88.h5')
        try:
            tensor_model = tf.keras.models.load_model('28_VGG16_88.h5', custom_objects={'tf': tf})
            st.write("TensorFlow Model Loaded Successfully!")
        except Exception as e:
            st.error(f"Error loading TensorFlow model: {e}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.4, 0.4, 0.4], [0.3, 0.3, 0.3])
    ])

    class StreamlitImageDataset(Dataset):
        def __init__(self, uploaded_files, transform=None):
            self.uploaded_files = uploaded_files
            self.transform = transform

        def __len__(self):
            return len(self.uploaded_files)

        def __getitem__(self, idx):
            image = Image.open(self.uploaded_files[idx]).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image

    def predict_with_torch(model, dataloader):
        model.eval()
        predictions = []
        with torch.no_grad():
            for img_tensor in dataloader:
                outputs = model(img_tensor)
                _, preds = torch.max(outputs, 1)
                predictions.append(preds.item())
        return predictions

    def rescale(image):
        image = tf.cast(image, tf.float32)
        image /= 255.0
        return tf.image.resize(image, [224, 224])

    def decode_image(uploaded_file):
        content = uploaded_file.read()
        img = tf.image.decode_jpeg(content, channels=3)
        img = rescale(img)
        img = tf.expand_dims(img, axis=0)  # Add batch dimension
        return img

    uploaded_images = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    col1, col2, col3 = st.columns([1,0.10,1])

    with col1:
        torch_prediction = st.button('Predict With Torch Model (Accuracy 97%)')
    with col3:
        tensor_prediction = st.button('Predict With VGG16 tf (Accuracy 88%)')

    if torch_prediction:
        if uploaded_images:
            dataset = StreamlitImageDataset(uploaded_files=uploaded_images, transform=transform)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
            predictions = predict_with_torch(model_torch, dataloader)
            for idx, (prediction, file) in enumerate(zip(predictions, uploaded_images)):
                st.image(file, caption=f'Predicted class: {classes_name[prediction]}', use_column_width=True)
        else:
            st.warning('No images were uploaded')

    if tensor_prediction:
        if uploaded_images:
            for image in uploaded_images:
                img = decode_image(image)
                prediction = tensor_model.predict(img)[0]  # No need for [0][0]
                predicted_class = np.argmax(prediction)  # Get the index of the highest probability
                st.image(image, caption=f'Predicted class: {classes_name[predicted_class]}', use_column_width=True)
        else:
            st.warning('No images were uploaded')
