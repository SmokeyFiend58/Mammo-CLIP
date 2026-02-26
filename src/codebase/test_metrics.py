import torch
from torch.utils.data import DataLoader
from MammoEval import MammoEval
from train_grading import VinDrSwinDataset, config
from breastclip.model.mammo_clip import MammoCLIP
from breastclip.data import data_utils
import pandas as pd
from train_grading import MultiHeadSwin

def testMain():
    args = config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    #setup data
    
    dataframe = pd.read_csv(args.csv_file)
    testDataframe = dataframe[dataframe['split']== 'test']
    
    #no augmentation
    tfms = data_utils.get_density_augmentation(args.img_size)
    
    test_dataset = VinDrSwinDataset(testDataframe, args.img_dir, transform_dict=tfms, split_group="test")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle = False, num_workers=0)
    
    print("loading model")
    
    #load weights
    checkpoint_path = "./output_swin/Swin_epoch_20.pth"
    if "Swin" in checkpoint_path:
        print("Loading Image only baseline")
        model = MultiHeadSwin(encoder_name=args.arch, img_size=args.img_size, density_loss_type=args.density_loss, birads_loss_type=args.birads_loss).to(device)
    else:
        print("Loading VLM")
        model = MammoCLIP(image_encoder_name=args.arch, img_size=args.img_size, use_aux_heads= False, use_uncertainty= False).to(device)
        
    
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict, strict=False)
    
    #run evalation
    evaluator = MammoEval(model, test_loader, device, output_path="./test_results")
    
    metrics = evaluator.evalMetrics()
    
    uncertainty = evaluator.evalUncertaintyMCDROPOUT(mc_samples=5)
    
if __name__ == "__main__":
    testMain()