from model import *
from data_loader import PoseDataset
import json
import pandas as pd
import torch
from utills import *
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import classification_report
from collections import Counter
from torch import nn
import sys
import copy


# get the model name from command line arg
model_input_name = sys.argv[1]


model_info = get_model_config(model_input_name)

#training info
num_workers = model_info["num_workers"]
num_epochs =  model_info["num_epochs"]
batch_size =  model_info["batch_size"]
learning_rate =  model_info["learning_rate"]
step_size =  model_info["step_size"]

#model_info
model_name = model_info["model_name"]
in_channels =  model_info["in_channels"]
num_bins_az =  model_info["num_bins_az"]
mask_size =  model_info["mask_size"]
num_bins_el =  model_info["num_bins_el"]
flag = model_info["flag"]

#data_loder_info
input_path =  model_info["input_path"]
features = model_info["added_feature"]
gt_D_mask_info_flag = model_info["gt_D_mask_info"]


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print ('device', device)



test_dict = eval(open("pix3d_s1_test.json",mode='r',encoding='utf-8').read())
test_im_list = pd.DataFrame(test_dict["images"])["file_name"]

train_dict = eval(open("pix3d_s1_train.json",mode='r',encoding='utf-8').read())
train_im_list = pd.DataFrame(train_dict["images"])["file_name"]

csv_file = open(input_path + "/Pix3D/Pix3D.txt")
data_f = pd.read_csv(csv_file)

# all info about all imgs in all categories
dict_pix3d = np.asarray(data_f)
raw_labels = pd.DataFrame(dict_pix3d[:,5:]) 

#D_mask_infor for each image(path to the right cad models)
gt_D_mask_info = raw_labels.set_index([0])
no_D_mask = None
#labels of all images

dd = generate_label(raw_labels, num_bins_az,num_bins_el)
labels  = dd[0]
overlap_label = dd[1]

train_dataset = PoseDataset(input_path, train_im_list, labels,overlap_label ,mask_size, features, eval(gt_D_mask_info_flag))
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
# feature, gt_D_mask_info
# train_dataset[0]

test_dataset = PoseDataset(input_path, test_im_list, labels,overlap_label, mask_size, features, eval(gt_D_mask_info_flag))
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


model = eval(model_name+"(in_channels, num_bins_az, mask_size, num_bins_el, flag)")
model.to(device)

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(list(model.parameters()), lr=learning_rate)
n_total_steps = len(train_dataloader)
scheduler = StepLR(optimizer, step_size=step_size)


for epoch in range(num_epochs):
        model.train()
        for i, (inputs, labels, cls,IDS,y_over,mask_path, mask_real) in enumerate(train_dataloader):
                
                features = inputs[0].to(device)
                mask = inputs[1].to(device)

                azimuth = labels[0].to(device)
                elevation = labels[1].to(device)
                
                model.train()

                optimizer.zero_grad()
                # compute the model output
                yhat = model(features,mask)
                # calculate loss
                train_loss = criterion(yhat[0], azimuth) + criterion(yhat[1], elevation)
                # credit assignment
                train_loss.backward()
                # update model weights
                optimizer.step()

        total = 0  
        total_el = 0  
        correct = 0 
        correct2 = 0  
        correct_el = 0 
        correct2_el = 0    
        model.eval()
        all_labels = []
        all_pred = []
        all_cls = []
        test_eq =0

        match = 0

        for i, (inputs, labels, cls,IDS,y_over,mask_path, mask_real) in enumerate(test_dataloader):

                features = inputs[0].to(device)
                mask = inputs[1].to(device)

                azimuth = labels[0].to(device)
                elevation = labels[1].to(device)


                # get top 3 of the model without the mask

                yhat = model(features, mask)

                
                optimizer.zero_grad()
                test_loss = criterion(yhat[0], azimuth) + criterion(yhat[1], elevation)
                
                _, predicted = torch.max(yhat[0].data, 1)
                _, predicted2 = torch.topk(yhat[0].data, 2, 1)
                
                _, predicted_el = torch.max(yhat[1].data, 1)
                _, predicted2_el = torch.topk(yhat[1].data, 2, 1)
                

                all_labels.extend(azimuth.cpu().tolist())
                all_pred.extend(predicted.cpu().tolist())
                all_cls.extend(list(cls))

                # Total number of labels
                total += azimuth.size(0)
                total_el += elevation.size(0)

                correct += torch.sum(predicted == azimuth).item()
                correct2 += torch.sum(torch.eq(predicted2, azimuth.reshape(-1,1))).item()

                correct_el += torch.sum(predicted_el == elevation).item()
                correct2_el += torch.sum(torch.eq(predicted2_el, elevation.reshape(-1,1))).item()
    
        accuracy = 100 * correct / total
        accuracy2 = 100 * correct2 / total

        accuracy_el = 100 * correct_el / total_el
        accuracy2_el = 100 * correct2_el / total_el
        
        print("############################################################################################")
        print (f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss.item():.4f}, Test Loss: {test_loss.item():.4f}, Val_Accuracy [{accuracy}], Val_Accuracy2 [{accuracy2}], Val_Accuracy [{accuracy_el}], Val_Accuracy2 [{accuracy2_el}]')
        # print (f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss.item():.4f}, Val_Accuracy [{accuracy}], Val_Accuracy2 [{accuracy2}]')
        # print(test_eq/total*25)
        print(classification_report(all_labels, all_pred))
        
        d2 = (Counter(np.array(all_cls)[np.array(all_labels) ==  np.array(all_pred)]))
        d1 = (Counter(all_cls))
        print(d1) 
        d3 = dict((k, "%.2f" % ( (float(d2[k]) / d1[k])*100 )) for k in d2)
        print(d3)

        scheduler.step()
        
print('Finished Training')
PATH = './model.pth'




















































# num_workers = 6
# num_epochs = 50
# batch_size = 10
# learning_rate = 0.001
# num_bins_az = 9
# num_bins_el = 5
# in_channels = 16
# step_size = 3
# input_path = "../../Datasets/pix3d"
# MODEL_PATH = "./model.pth"
# mask_size =128

# #added_features = ['autoencoding','depth_euclidean','jigsaw' ,'reshading','colorization',
# #'edge_occlusion','keypoints2d','room_layout',
# #'curvature'  ,'keypoints3d'  ,'segment_unsup2d'  ,
# #'class_object' ,'egomotion' ,  'nonfixated_pose'   , 'segment_unsup25d',
# #'class_scene',  'fixated_pose'  , 'segment_semantic',      
# #'denoising' , 'inpainting'   ,'point_matching' ,   'vanishing_point'
# #]

# #features = ['jigsaw']


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print ('device', device)

# model_D_mask_reduction = PoseEstimationModelUpsampel_V1_MaskedFeatures(in_channels, num_bins_az, mask_size,num_bins_el)
# model_D_mask_reduction.load_state_dict(torch.load(MODEL_PATH))
# # model_D_mask_reduction = nn.DataParallel(model_D_mask_reduction)
# model_D_mask_reduction.eval()
# model_D_mask_reduction.to(device)

# #for feature in features:

# test_dict = eval(open("pix3d_s1_test.json",mode='r',encoding='utf-8').read())
# test_im_list = pd.DataFrame(test_dict["images"])["file_name"]

# train_dict = eval(open("pix3d_s1_train.json",mode='r',encoding='utf-8').read())
# train_im_list = pd.DataFrame(train_dict["images"])["file_name"]

# csv_file = open(input_path + "/Pix3D/Pix3D.txt")
# data_f = pd.read_csv(csv_file)
# # all infor about all imgs in all categories
# dict_pix3d = np.asarray(data_f)
# raw_labels = pd.DataFrame(dict_pix3d[:,5:]) 

# gt_D_mask_info = raw_labels.set_index([0])
# labels  = generate_label(raw_labels, num_bins_az,num_bins_el)

# #print("The "+ feature + " has been added.")
# train_dataset = PoseDataset(input_path, train_im_list, labels ,mask_size,None,gt_D_mask_info)
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
# # train_dataset[0]

# test_dataset = PoseDataset(input_path,test_im_list, labels,mask_size,None,gt_D_mask_info)
# test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

# model = resnet_net(in_channels, num_bins_az, mask_size)
# # model = nn.DataParallel(model)

# # model = PoseEstimationModelUpsampel_V1_MaskAsChannel(in_channels, num_bins, mask_size)
# # model = PoseEstimationModel_baseline(in_channels, num_bins)

# model.to(device)

# criterion = torch.nn.CrossEntropyLoss()

# optimizer = torch.optim.Adam(list(model.parameters()), lr=learning_rate)
# n_total_steps = len(train_dataloader)
# scheduler = StepLR(optimizer, step_size=step_size)

# match_test_cad = json.load(open("match.json",))
# top_num = 3
# for epoch in range(num_epochs):
#         model.train()
#         for i, (inputs, labels, cls,IDS,mask_path, img_name, mask_real, mask_gt,img_raw) in enumerate(train_dataloader):
#                 size_batch = len(labels[0])
#                 D_masks = torch.zeros((len(labels[0]),top_num,1,mask_size,mask_size)).to(device)
#                 features = inputs[0].to(device)
#                 mask = inputs[1].to(device)

#                 azimuth = labels[0].to(device)
#                 elevation = labels[1].to(device)
                

#                 y_hat_top = model_D_mask_reduction(features, mask,0)

#                 _, predicted2 = torch.topk(y_hat_top[0].data, top_num, 1)
#                 _, predicted2_el = torch.topk(y_hat_top[1].data, 2, 1)

#                 a = np.arange(size_batch*top_num)
#                 np.random.shuffle(a)
#                 a = a.reshape((size_batch,top_num))
#                 a = a.argsort()

#                 D_masks[:,0] = get_Dmask(predicted2[:,0],elevation,IDS,gt_D_mask_info).to(device).to(device)
#                 D_masks[:,1] = get_Dmask(predicted2[:,1],elevation,IDS,gt_D_mask_info).to(device).to(device)
#                 D_masks[:,2] = get_Dmask(predicted2[:,2],elevation,IDS,gt_D_mask_info).to(device).to(device)
#                 # D_masks[:,3] = get_Dmask(predicted2[:,3],elevation,IDS,gt_D_mask_info).to(device).to(device)


#                 D_masks = D_masks[torch.arange(D_masks.shape[0]).reshape(D_masks.shape[0],1),a]

#                 yhat = model(features,D_masks[:,0],D_masks[:,1], D_masks[:,2], mask_gt.reshape(size_batch,1,mask_size,mask_size).to(device),img_raw.to(device).float())
#                 # ,D_masks[:,2],D_masks[:,3]
#                 optimizer.zero_grad()
#                 # compute the model output
#                 # calculate loss
#                 train_loss = criterion(yhat, azimuth) 
#                 # credit assignment
#                 train_loss.backward()
#                 # update model weights
#                 optimizer.step()

#         total = 0  
#         total_el = 0  
#         correct = 0 
#         correct2 = 0  
#         correct_el = 0 
#         correct2_el = 0    
#         model.eval()
#         all_labels = []
#         all_pred = []
#         all_cls = []
#         test_eq =0

#         match = 0

#         for i, (inputs, labels, cls,IDS,mask_path, img_name, mask_real, mask_gt,img_raw)  in enumerate(test_dataloader):
                

#                 size_batch = len(labels[0])
#                 D_masks = torch.zeros((len(labels[0]),top_num,1,mask_size,mask_size)).to(device)

#                 features = inputs[0].to(device)
#                 mask = inputs[1].to(device)

#                 azimuth = labels[0].to(device)
#                 elevation = labels[1].to(device)


#                 # get top 3 of the model without the mask

#                 y_hat_top = model_D_mask_reduction(features, mask,0)

#                 _, predicted2 = torch.topk(y_hat_top[0].data, top_num, 1)
#                 _, predicted2_el = torch.topk(y_hat_top[1].data, 2, 1)

#                 match_ID = []
#                 for keys in IDS:
#                         match_ID.append("img/"+match_test_cad[keys])

#                 a = np.arange(size_batch*top_num)
#                 np.random.shuffle(a)
#                 a = a.reshape((size_batch,top_num))
#                 a = a.argsort()

#                 D_masks[:,0] = get_Dmask(predicted2[:,0],predicted2_el[:,0],IDS,gt_D_mask_info).to(device).to(device)
#                 D_masks[:,1] = get_Dmask(predicted2[:,1],predicted2_el[:,0],IDS,gt_D_mask_info).to(device).to(device)

#                 D_masks[:,2] = get_Dmask(predicted2[:,2],predicted2_el[:,0],IDS,gt_D_mask_info).to(device).to(device)
#                 # D_masks[:,3] = get_Dmask(predicted2[:,3],predicted2_el[:,0],match_ID,gt_D_mask_info).to(device).to(device)


#                 yhat = model(features,D_masks[:,0],D_masks[:,1], D_masks[:,2],mask_real.to(device),img_raw.to(device).float())
#                 # D_masks[:,3],
#                 # optimizer.zero_grad()

#                 # all_labels.extend(azimuth.cpu().tolist())
#                 #all_pred.extend(predicted.cpu().tolist())
#                 # all_cls.extend(list(cls))

#                 # Total number of labels
#                 total += azimuth.size(0)
#                 total_el += elevation.size(0)

#                 #correct += torch.sum(output_eq_pred_dmask).item()
#                 #print(correct)
#                 prob_azimuth, predicted = torch.max(yhat.data, 1)

#                 false_results = (predicted.cpu() != azimuth.cpu())

#                 if(epoch == 4):
#                         for i in range(len(false_results)):
#                                 if (false_results[i]):
#                                         save_top_masks(mask_path[i], img_name[i], predicted[i], predicted2_el[i,0],prob_azimuth[i],match_ID[i],gt_D_mask_info)

                

#                 correct += torch.sum(predicted.cpu() == azimuth.cpu()).item()
#                 #correct2 += torch.sum(torch.eq(predicted2, azimuth.cpu().reshape(-1,1))).item()

#                 # correct_el += torch.sum(predicted_el == elevation).item()
#                 # correct2_el += torch.sum(torch.eq(predicted2_el, elevation.reshape(-1,1))).item()
    
#         accuracy = 100 * correct / total
#         accuracy2 = 100 * correct2 / total

#         # accuracy_el = 100 * correct_el / total_el
#         # accuracy2_el = 100 * correct2_el / total_el
        
#         print("############################################################################################")
#         #print (f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss.item():.4f}, Test Loss: {test_loss.item():.4f}, Val_Accuracy [{accuracy}], Val_Accuracy2 [{accuracy2}], Val_Accuracy [{accuracy_el}], Val_Accuracy2 [{accuracy2_el}]')
#         print (f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss.item():.4f}, Val_Accuracy [{accuracy}], Val_Accuracy2 [{accuracy2}]')
#         print(test_eq/total*25)
#         #print(classification_report(all_labels, all_pred))
        
#         #d2 = (Counter(np.array(all_cls)[np.array(all_labels) ==  np.array(all_pred)]))
#         #d1 = (Counter(all_cls))
#         #print(d1) 
#         #d3 = dict((k, "%.2f" % ( (float(d2[k]) / d1[k])*100 )) for k in d2)
#         #print(d3)

#         scheduler.step()
        
# print('Finished Training')
# PATH = './model.pth'
# #torch.save(model.state_dict(), PATH)



