# **DSGAN in Semi-Supervised Learning**

## **Dataset**

MNIST <br/> 
SVHN <br/> 
CIFAR-10 <br/> 

## **Requirement**

python >= 3.5 <br/>
pytorch == 1.2.0 <br/>
torchvision == 0.4.0 <br/>

## **Execution**

### **training**

Train MNIST <br/>

`bash train_mnist.sh`

Train SVHN <br/>

`bash train_svhn.sh`

Train CIFAR-10 <br/>

`bash train_cifar.sh`

Note the most of the parameters shown in the bash file follow other's work, the only important parameters for our method are `alpha` and `beta_2`.

