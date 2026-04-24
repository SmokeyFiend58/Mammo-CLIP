import torch
import os
from torch.utils.data import DataLoader
from src.codebase.MammoEval import MammoEval

from src.codebase.breastclip.model.mammo_clip import MammoCLIP
from src.codebase.breastclip.data import data_utils
import pandas as pd
from src.codebase.train_grading import MultiHeadSwin, VinDrSwinDataset, config

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
    #replaced to not be hardcoded
    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No checkpoint at the path - pass --checkpoint to overide"
        )
    
    if "swin" in checkpoint_path.lower():
        print("Loading Image only baseline")
        model = MultiHeadSwin(encoder_name=args.arch, img_size=args.img_size, density_loss_type=args.density_loss, birads_loss_type=args.birads_loss).to(device)
    else:
        print("Loading VLM")
        image_encoder = args.image_encoder or args.arch
        
        model = MammoCLIP(image_encoder_name=args.arch,text_encoder_name =args.text_encoder, img_size=args.img_size, use_aux_heads= True, use_uncertainty= True).to(device)
        
    
    state_dict = torch.load(checkpoint_path, map_location = device)
    model.load_state_dict(state_dict, strict=False)
    
    #run evalation
    evaluator = MammoEval(model, test_loader, device, output_path="./test_results", density_loss = args.density_loss, birads_loss = args.birads_loss)
    
    
    metrics = evaluator.evalMetrics()
    
    uncertainty = evaluator.evalUncertaintyMCDROPOUT(mc_samples=5)
    
if __name__ == "__main__":
    testMain()