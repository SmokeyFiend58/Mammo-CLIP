import pandas as pd
import json
import os

# handles metrics hopefully making it a bit cleaner
class ResultsLogger:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.history = []
        os.makedirs(output_dir, exist_ok=True)
    
    def log_epoch(self, epoch, metrics):
        
        metrics['epoch'] = epoch
        self.history.append(metrics)
        
        #save incremental csv
        
        dataFrame = pd.DataFrame(self.history)
        dataFrame.to_csv(os.path.join(self.output_dir, "training_log.csv"), index=False)
        
    def saveFinalResult(self, args, final_metrics):
        #save summary json with config + final results for nice comparison
        result = {"config": vars(args), "results": final_metrics}
        
        with open(os.path.join(self.output_dir, "final_results.json"), "w") as f:
            json.dump(result, f, indent=4)
            