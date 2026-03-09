import os, torch, torch.nn as nn, torch.optim as optim, torchvision.transforms as transforms, torchvision.models as models
from torch.utils.data import DataLoader, random_split, TensorDataset;from experiment_template import *;from tqdm import tqdm;import pandas as pd
def main():
    device,data_root,num_classes="cuda" if torch.cuda.is_available() else "cpu","./tnn2025",46
    train_csv,train_img_dir,test_img_dir=os.path.join(data_root,"train.csv"),os.path.join(data_root,"train/train_256"),os.path.join(data_root,"test/test_256")
    output_dir="./features_effb2";os.makedirs(output_dir,exist_ok=True)
    train_path,test_path=os.path.join(output_dir,"train_effb2.pt"),os.path.join(output_dir,"test_effb2.pt")
    if os.path.exists(train_path)and os.path.exists(test_path):print("\n--- Loading efficientnet_b2 Features ---");train_data,test_data=torch.load(train_path,weights_only=True),torch.load(test_path,weights_only=True)
    else:print("\n--- Run feature extraction first ---");return
    print("\n--- Training efficientnet_b2 + TransformerHead ---")
    full_ds=TensorDataset(train_data["features"],train_data["labels"]);ts=int(0.9*len(full_ds));t_db,v_db=random_split(full_ds,[ts,len(full_ds)-ts])
    t_ld,v_ld=DataLoader(t_db,batch_size=128,shuffle=True),DataLoader(v_db,batch_size=128,shuffle=False)
    model=TransformerHead(input_dim=1408,num_classes=num_classes).to(device)
    optimizer=optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-2);criterion,num_epochs,best_acc=nn.CrossEntropyLoss(),50,0
    for ep in range(1,num_epochs+1):
        model.train()
        for x,y in t_ld:x,y=x.to(device),y.to(device);optimizer.zero_grad();loss=criterion(model(x),y);loss.backward();optimizer.step()
        model.eval();corr=0
        with torch.no_grad():
            for x,y in v_ld:_,p=model(x.to(device)).max(1);corr+=p.eq(y.to(device)).sum().item()
        acc=corr/len(v_db)
        if acc>best_acc:best_acc=acc;torch.save(model.state_dict(),"efficientnet_b2_transformer.pth")
        print(f"Epoch {ep}/{num_epochs} | Val Acc: {acc:.4f} (Best: {best_acc:.4f})")
    print("\n--- Generating Submission ---");model.load_state_dict(torch.load("efficientnet_b2_transformer.pth",weights_only=True));model.eval();all_preds=[]
    with torch.no_grad():
        for i in range(0,len(test_data["features"]),128):_,p=model(test_data["features"][i:i+128].to(device)).max(1);all_preds.extend(p.cpu().tolist())
    pd.DataFrame({"id":test_data["labels"].tolist(),"label":all_preds}).sort_values("id").to_csv("submission_effb2_trans.csv",index=False);print("Done!")
if __name__=="__main__":main()
