# **DSGAN in Novelty Detection**

## **Dataset**

CIFAR-10 <br/> 

## **Requirement**

python >= 3.5 <br/>
pytorch == 0.4.1 <br/>
torchvision == 0.2.1 <br/>
scikit-learn <br/>
matplotlib <br/>

## **Execution**

### **training**

Train the models in three steps:

`bash train_all.sh`

Note that there are three-steps training in the base file. However, one can simply run any one of three steps by commenting out codes for other steps. 

### **testing**

Test the models which save in `model/cifar_finetune_vae/vae_checkpoint.pt`

`bash test.sh`

The AUCs will be recorded in `log/cifar_finetune_vae/record.txt`
