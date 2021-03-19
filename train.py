import torch
import utils
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
from torch.utils.data import DataLoader
import modules
import time
import numpy as np
import os
import shutil

def train_meta(model,train_dataloader,val_dataloader,epochs,lr,steps_til_summary,epochs_til_checkpoint,model_dir,outer_loss_fn, summary_fn,add_noise=0,lr_type=None,gradient_type=None,clip=True,tunedenoiser=False,xyencoder=False):
    
    if tunedenoiser and xyencoder:
        optim = torch.optim.Adam([{"params":model.siren.parameters(),'lr':lr},
                                {"params":model.lr.parameters(),'lr':lr},
                                {"params":model.gradient_steps,'lr':5e-3},
                                {"params":model.position_encoder,'lr':2e-3}])
    elif tunedenoiser:
        optim = torch.optim.Adam(lr=lr, params=model.parameters(), amsgrad=False)
    # else:
    #     optim = torch.optim.Adam([{"params":model.siren.parameters(),'lr':lr},
    #                             {"params":model.lr.parameters(),'lr':lr},
    #                             {"params":model.gradient_steps,'lr':5e-3}])
    
    # if not (gradient_type=="static" or lr_type=="static"):
    #     optim = torch.optim.Adam([{"params":model.siren.parameters(),'lr':lr},
    #                             {"params":model.lr.parameters(),'lr':1e-4},
    #                             {"params":model.gradient_steps,'lr':5e-3}])
    # elif not (gradient_type=="static"):
    #     optim = torch.optim.Adam([{"params":model.siren.parameters(),'lr':lr},
    #                             {"params":model.gradient_steps,'lr':5e-4}])
    # elif not (lr_type=="static"):
    #     optim = torch.optim.Adam([{"params":model.siren.parameters(),'lr':lr},
    #                             {"params":model.lr,'lr':1e-4}])
    # else:
    #     optim = torch.optim.Adam(lr=lr, params=model.parameters())

    optim = torch.optim.Adam(lr=lr, params=model.parameters(), amsgrad=False)

    summaries_dir = os.path.join(model_dir, 'summaries')
    utils.cond_mkdir(summaries_dir)

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    utils.cond_mkdir(checkpoints_dir)

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print (f'\n\nTraining model with {num_parameters} parameters\n\n')

    writer = SummaryWriter(summaries_dir)
    
    # checkpoints_dir2 = os.path.join("/media/data2/cyanzhao/meta_solve/debluring_ircnn_fp/ablation2_sigma7.65div2/resol64_faceTrue_csolvergd_ks5_sig1.5_noise0.030000000000000002_lr1e-05_innerlr1_innerlrtypper_parameter_nummeta3_numsolver3_solverlr1_data150000_samplesmooth_batch3_tunedenoiserTrue_xyencoderFalse_denoiserdncnn_note90_l2", 'checkpoints')
    # PATH = "{}/model_current.pth".format(checkpoints_dir2)
    # checkpoint = torch.load(PATH,map_location='cpu')
    # checkpoint = checkpoint['model_state_dict']
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optim.load_state_dict(checkpoint['optimizer_state_dict'])

    total_steps = 0
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []
        for epoch in range(epochs):
            for step, task_batch in enumerate(train_dataloader):

                if not (step%1000) and step>0:
                    torch.save({'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict':optim.state_dict(),
                                'loss':outer_losses}
                                ,os.path.join(checkpoints_dir,"model_current_backup.pth"))

                start_time = time.time()
                model_outputs, gts, observations, inner_losses,_,_= model(task_batch,add_noise)
                #model_outputs: list of outputs for for different tasks in the batch 
                #gts: ground truth value for all the taskes
                #inner_losses: average of inner loss for each updates
                outer_losses = outer_loss_fn(model_outputs=model_outputs,gts=gts,observation=observations)
                #outer_losses: average loss for all taskes in the task_batch
                outer_loss = 0
                for loss_name, loss in outer_losses.items():
                    single_loss = loss.mean()
                    outer_loss+=single_loss

                train_losses.append(outer_loss.detach().cpu().numpy())
                writer.add_scalar("train_loss",outer_loss,total_steps)
                # writer.add_scalar("inner_loss",inner_losses[-1],total_steps)
                
                nanflag = False

                if torch.isnan(outer_loss) or outer_loss>1 or torch.sum(torch.isnan(inner_losses)*1)>0:
                    nanflag=True
                    torch.save(outer_loss,'{}/outer_loss.pt'.format(checkpoints_dir))
                    torch.save(gts,'{}/gts.pt'.format(checkpoints_dir))
                    torch.save(observations,'{}/observations.pt'.format(checkpoints_dir))
                    torch.save(inner_losses,'{}/inner_losses.pt'.format(checkpoints_dir))
                    torch.save(model_outputs,'{}/model_outputs.pt'.format(checkpoints_dir))
                    print("detect NaN")
                    # print(outer_loss)
                    # print(gts)
                    # print(observations)
                    # print(inner_losses)
                    # print(model_outputs)
                    

                optim.zero_grad()
                # start_backward_time = time.time()
                outer_loss.backward()

                
                if clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

                # for parameter in model.siren.parameters():
                #         if torch.sum(torch.isnan(parameter.grad)*1)>0:
                #             nanflag = True
                if not nanflag:
                    optim.step()
                pbar.update(1)
                # finishupdate = time.time()-start_backward_time
                # print("finish meta-update{}".format(finishupdate))
                
                #####evaluation and summary######
                if not total_steps % steps_til_summary:
                    # tqdm.write(
                    # "Epoch %d, inner loss 0: %0.6f, inner loss -1: %0.6f, outer loss %0.6f, iteration time %0.6f" %
                    # (epoch, inner_losses[0], inner_losses[-1], outer_loss, time.time() - start_time))
                    tqdm.write("Epoch %d, outer loss %0.6f, iteration time %0.6f" % (epoch, outer_loss, time.time() - start_time))
                    # for name,parameters in model.siren.named_parameters():
                    #     print(name + ": {}".format(parameters.grad.mean()))
                    summary_fn(gt=gts, observation=observations, model_outputs=model_outputs, writer=writer,total_steps=total_steps,prefix="train")

                    if model.siren!=None:
                        for name, parameter in model.siren.named_parameters():
                            writer.add_histogram("train" + name, parameter.cpu(), global_step=total_steps)
                            writer.add_histogram("train" + name + "grad", parameter.grad.cpu(), global_step=total_steps)
                    if tunedenoiser:
                        for name, parameter in model.denoiser.named_parameters():
                            writer.add_histogram("train" + name, parameter.cpu(), global_step=total_steps)
                            if (parameter.grad!=None):
                                writer.add_histogram("train" + name + "grad", parameter.grad.cpu(), global_step=total_steps)

                    if xyencoder:
                        writer.add_histogram("xyencoder", model.position_encoder.cpu(), global_step=total_steps)
                    # writer.add_scalar("train_gradient_step",model.gradient_steps,global_step=total_steps)

                    torch.save({'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict':optim.state_dict(),
                                'loss':outer_losses}
                                ,os.path.join(checkpoints_dir,"model_current.pth"))
                    if not (val_dataloader==None):
                        print("running evaluation")
                        model.eval()
                        val_losses = []
                        for val_idx, val_task_batch in enumerate(val_dataloader):
                            model_outputs, gts, observations, inner_losses,_,_= model(val_task_batch,add_noise,mode="eval")
                            outer_loss = outer_loss_fn(model_outputs=model_outputs,gts=gts,observation=observations)
                            val_loss = 0
                            for loss_name, loss in outer_loss.items():
                                single_loss = loss.mean()
                                val_loss+=single_loss.detach().cpu().numpy()

                            val_losses.append(val_loss)

                            summary_fn(gt=gts, observation=observations, model_outputs=model_outputs, writer=writer,total_steps=total_steps,prefix="val")

                        writer.add_scalar("val_loss",np.mean(val_losses),total_steps)
                        tqdm.write("Validation loss %0.6f" % np.mean(val_losses))
                        model.train()
                total_steps += 1

                if not epoch % epochs_til_checkpoint:
                    torch.save(model.state_dict(),
                           os.path.join(checkpoints_dir, 'epoch_%03d.pth'%epoch))
                    np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%03d.txt'%epoch),
                           np.array(train_losses))
    torch.save(model.state_dict(),
                   os.path.join(checkpoints_dir, 'model_final.pth'))
    np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
                   np.array(train_losses))

                

