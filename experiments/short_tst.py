from ConcatModels import SAM_MetaC
from Postprocess import load_checkpoint
from Preprocess import process_img
from Visualize import show_pred
from Dataset import add_btch_dim

sam = SAM_MetaC(in_channel_count=3, img_size=1024, num_class=6)

sam:SAM_MetaC = load_checkpoint(
    model=sam,
    model_path='/nas-ctm01/homes/mhzolfagharnasab/segmentation-framework/artifacts/binary/model/SAM_MetaC_basic.pth',
    device='cpu'
)

image_dict = add_btch_dim(process_img(img_input='/nas-ctm01/homes/mhzolfagharnasab/segmentation-framework/notebooks/51346_4nov2016_002.JPG', trg_size=1024))

pred = sam.predict_one(image_dict)

show_pred(img_dict=image_dict, 
          prd_msk=pred, 
          save_dir='/nas-ctm01/homes/mhzolfagharnasab/segmentation-framework/notebooks/', 
          name='51346_4nov2016_002_tst.JPG')

