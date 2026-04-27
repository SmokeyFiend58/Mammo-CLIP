import torch
import os
import re
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

    #filter out BI-RADS 0 (incomplete assessment, not a final grade) to match training distribution
    def extractBIRADS(raw):
        if pd.isnull(raw):
            return None
        m = re.search(r"[0-6]", str(raw))
        return int(m.group(0)) if m else None

    before = len(dataframe)
    dataframe['_birads_int'] = dataframe['breast_birads'].apply(extractBIRADS)
    dataframe = dataframe[dataframe['_birads_int'].between(1, 5)].drop(columns=['_birads_int'])
    print(f"Filtered BI-RADS outside 1-5 (incl. 0/6/NaN): {before} -> {len(dataframe)} rows")

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
    
    state_dict = torch.load(checkpoint_path, map_location = device)

    if "swin" in checkpoint_path.lower():
        print("Loading Image only baseline")
        model = MultiHeadSwin(encoder_name=args.arch, img_size=args.img_size, density_loss_type=args.density_loss, birads_loss_type=args.birads_loss).to(device)
    else:
        print("Loading VLM")
        image_encoder = args.image_encoder or args.arch

        #infer embed_dim from the saved projection weight, so we don't
        #have to remember which run used 256 vs 512.
        if "image_projection.weight" in state_dict:
            embed_dim = state_dict["image_projection.weight"].shape[0]
        else:
            embed_dim = 512
        print(f"  inferred embed_dim from checkpoint: {embed_dim}")

        #infer use_uncertainty from the density-percentage head shape.
        #if it has 2 outputs the run trained the (mu, log var) Gaussian-NLL head;
        #if it has 1 output the run used plain MSE.
        #aux heads themselves are presumed present for any cell we test.
        has_aux = "head_density_class.weight" in state_dict
        if "head_density_perc.weight" in state_dict:
            use_unc = state_dict["head_density_perc.weight"].shape[0] == 2
        else:
            use_unc = False
        print(f"  inferred use_aux_heads={has_aux}, use_uncertainty={use_unc}")

        model = MammoCLIP(image_encoder_name=args.arch,text_encoder_name =args.text_encoder, img_size=args.img_size, embed_dim=embed_dim, use_aux_heads= has_aux, use_uncertainty= use_unc).to(device)

    model.load_state_dict(state_dict, strict=False)
    
    #run evalation
    evaluator = MammoEval(model, test_loader, device, output_path="./test_results", density_loss = args.density_loss, birads_loss = args.birads_loss)
    
    
    metrics = evaluator.evalMetrics()
    
    uncertainty = evaluator.evalUncertaintyMCDROPOUT(mc_samples=5)
    
if __name__ == "__main__":
    testMain()