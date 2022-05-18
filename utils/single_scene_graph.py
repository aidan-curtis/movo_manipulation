# System libs
import os, csv, torch, numpy, scipy.io, PIL.Image, torchvision.transforms
# Our libs
from semseg.mit_semseg.models import ModelBuilder, SegmentationModule
from semseg.mit_semseg.utils import colorEncode
import numpy as np

colors = scipy.io.loadmat('semseg/data/color150.mat')['colors']
names = {}
with open('semseg/data/object150_info.csv') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        names[int(row[0])] = row[5].split(";")[0]

def visualize_result(img, pred, index=None):
    # filter prediction class if requested
    if index is not None:
        pred = pred.copy()
        pred[pred != index] = -1
        #print(f'{names[index+1]}:')
        
    # colorize prediction
    pred_color = colorEncode(pred, colors).astype(numpy.uint8)

    # aggregate images and save
    pil_im = PIL.Image.fromarray(pred_color)
    pil_im.show()


# Network Builders
net_encoder = ModelBuilder.build_encoder(
    arch='resnet50dilated',
    fc_dim=2048,
    weights='semseg/ckpt/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth')
net_decoder = ModelBuilder.build_decoder(
    arch='ppm_deepsup',
    fc_dim=2048,
    num_class=150,
    weights='semseg/ckpt/ade20k-resnet50dilated-ppm_deepsup/decoder_epoch_20.pth',
    use_softmax=True)

crit = torch.nn.NLLLoss(ignore_index=-1)
segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
segmentation_module.eval()
segmentation_module.cuda()


# Load and normalize one image as a singleton tensor batch
pil_to_tensor = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406], # These are RGB mean+std values
        std=[0.229, 0.224, 0.225])  # across a large photo dataset.
])

label_dict = { 0:0, # Walls, 
            3:3, # floor 
            5:5, # ceiling 
            15:15, # table
            56:15,
            64:15,
            33:15,
            19:19, # chair
            30:19,
            75:19,
            31:19}

def get_semantic_labels(img_data):
    np_image = numpy.array(img_data).astype(numpy.uint8)
    # np_image = numpy.transpose(np_image, (2, 0, 1))
    #print(np_image.shape)
    pil_image = PIL.Image.fromarray(np_image)
    img_data = pil_to_tensor(pil_image)

    singleton_batch = {'img_data': img_data[None].cuda()}
    output_size = img_data.shape[1:]


    # Run the segmentation at the highest resolution.
    with torch.no_grad():
        scores = segmentation_module(singleton_batch, segSize=output_size)
        
    # Get the predicted scores for each pixel
    _, pred = torch.max(scores, dim=1)
    pred = pred.cpu()[0].numpy()
    #print(pred)

    # # filter prediction class if requested
    pred = pred.copy()
    zero_pred = np.ones(pred.shape)*-1

    for index in label_dict.keys():
        zero_pred[pred == index] = label_dict[index]
        #print(f'{names[index+1]}:')
            
    pred = zero_pred
    # colorize prediction
    pred_color = colorEncode(pred, colors).astype(numpy.uint8)
    return pred_color

if __name__=="__main__":
    pil_image = PIL.Image.open('rgb/1649195040065.png').convert('RGB')
    img_original = numpy.array(pil_image)
    img_data = pil_to_tensor(pil_image)
    #print(img_data)
    singleton_batch = {'img_data': img_data[None].cuda()}
    output_size = img_data.shape[1:]


    # Run the segmentation at the highest resolution.
    with torch.no_grad():
        scores = segmentation_module(singleton_batch, segSize=output_size)
        
    # Get the predicted scores for each pixel
    _, pred = torch.max(scores, dim=1)
    pred = pred.cpu()[0].numpy()
    visualize_result(img_original, pred)
