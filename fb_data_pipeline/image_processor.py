from clean_data import CleanImage
import os
from torchvision import transforms
from PIL import Image

image_path = os.path.join(os.path.dirname(os.getcwd()), 'test_image')
image_cleaner = CleanImage(image_path)
cleaned_img_folder = image_cleaner.process_images(resized_pixel=(128, 128), mode = 'RGB', replace = True)
convert_tensor = transforms.ToTensor()
img = Image.open('/Users/serenawong/Desktop/fb_marketplace/ffdbe092-377b-4e90-8437-1faa964e5bf4.jpg')
img_tensor = convert_tensor(img)
img_numpy = img_tensor.numpy()
img_numpy = np.expand_dims(img_numpy, axis=0)
img_torch = torch.from_numpy(img_numpy)