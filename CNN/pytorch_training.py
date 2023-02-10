import torch as th
from tqdm import tqdm

import numpy as np


########################
class train_history():
    def __init__(self, train_epochs):
        self.history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        self.params = {'verbose': 1, 'epochs': train_epochs, 'steps': None}

    def save_history_in_this_epoch(self, train_loss, train_acc, val_loss, val_acc):
        self.history['loss'].append(train_loss)
        self.history['accuracy'].append(train_acc)
        self.history['val_loss'].append(val_loss)
        self.history['val_accuracy'].append(val_acc)

class train_setup:
    def __init__(self, model, optimizer, criterion, device, patience=5, save_model_name='.'):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.save_model_name = save_model_name
        self.patience = patience

        self.no_better_result = 0
        self.min_val_loss = np.inf
        self.terminate = False

        self.test_model = None
        self.test_predictions = []
        self.test_labels = []

        #self.model.finalized_kappa = th.tensor([-1]).to(self.device)
        
    def early_stopping(self):
        self.terminate = True

    def train_each_epoch(self, epoch, train_loader, train_epochs):
        train_loss_one = 0
        kappa_one = 0
        N_train = 0
        correct_train = 0
        loop = tqdm(enumerate(train_loader), total=len(train_loader), ncols=150)
        for i, (images, images2, pTnorms, labels) in loop:
            images = (images).to(self.device)
            images2 = (images2).to(self.device)
            pTnorms = (pTnorms).to(self.device)
            labels = (labels).to(self.device)
            
            #print (images.dtype)
            
            # init optimizer
            self.optimizer.zero_grad()
            outputs = self.model(images, images2, pTnorms)
            #print (outputs)
            loss = self.criterion(outputs, labels)
            #print (loss)
            train_loss_one += loss.item() * labels.size(0)
            
            loss.backward()
            self.optimizer.step()
            N_train += labels.size(0)

            kappa = self.model.kappa_transformation(self.model.kappa0)
            kappa_one += kappa.item() * labels.size(0)
            self.model.finalized_kappa = th.tensor([kappa_one/ N_train]).to(self.device)

            #print (outputs)
            pred = outputs.data.argmax(dim=1).float()
            #correct_train += (th.equal(pred,labels)).float().sum()
            correct_train += (pred == labels.argmax(dim=1)).float().sum()
            acc = 100 * correct_train / float(N_train)

            loop.set_description(f'Epoch [{epoch+1:3d}/{train_epochs}] | train loss= {train_loss_one/N_train:.3f}, train acc= {acc:.2f}% / kappa= {kappa_one / N_train:.5f} |')
        
        train_acc = (acc.item())
        train_loss = (loss.item())

        return train_loss, train_acc, outputs


    def valid_each_epoch(self, epoch, valid_loader):
        valid_loss_one = 0
        N_valid = 0
        correct_valid = 0
        for i, (images, images2, pTnorms, labels) in enumerate(valid_loader):
            #send data to cpu or gpu
            images = (images).to(self.device)
            images2 = (images2).to(self.device)
            pTnorms = (pTnorms).to(self.device)
            labels = (labels).to(self.device)

            # calculate model 
            outputs = self.model(images, images2, pTnorms)
            loss = self.criterion(outputs, labels)
            valid_loss_one += loss.item() * labels.size(0)

            N_valid += labels.size(0)

            pred = outputs.data.argmax(dim=1).float()
            correct_valid += (pred == labels.argmax(dim=1)).float().sum()
            acc = 100 * correct_valid / float(N_valid)
        print (f'VALIDATION:   kappa = {self.model.finalized_kappa.item() :.5f}     | val loss= {loss.item():.3f}, val acc= {acc.item():.2f}% |')
        
        valid_acc = (acc.item())
        valid_loss = (loss.item())

        # determine whether to archive the model (minized val_loss)
        if (self.min_val_loss>valid_loss):
            self.no_better_result = 0
            print (f'Epoch {epoch+1}: val_loss improved from {self.min_val_loss:.4f} to {valid_loss:.4f}, saving model to'+self.save_model_name)
            self.min_val_loss = valid_loss
            th.save(self.model.state_dict(), self.save_model_name)
        else:
            self.no_better_result += 1
            print (f'Epoch {epoch+1:3d}: val_loss did not improve from {self.min_val_loss:.4f}. Performance did not improve for {self.no_better_result:2d} epoch(s)')

        if (self.no_better_result>=self.patience):
            self.early_stopping()

        return valid_loss, valid_acc

    def test(self, test_loader):
        test_loss_one = 0
        N_test = 0
        correct_test = 0
        for i, (images, images2, pTnorms, labels) in enumerate(test_loader):
            #send data to cpu or gpu
            images = (images).to(self.device)
            images2 = (images2).to(self.device)
            pTnorms = (pTnorms).to(self.device)
            labels = (labels).to(self.device)

            # calculate model 
            outputs = self.test_model(images, images2, pTnorms)
            loss = self.criterion(outputs, labels)
            test_loss_one += loss.item() * labels.size(0)

            N_test += labels.size(0)

            pred = outputs.data.argmax(dim=1).float()
            correct_test += (pred == labels.argmax(dim=1)).float().sum()
            prob_preds = outputs.data
            #print (i, "pred", pred)
            #self.test_predictions.extend(prob_preds.cpu().numpy())
            self.test_predictions.extend(outputs.data.cpu().numpy())
            self.test_labels.extend(labels.cpu().numpy())


            acc = 100 * correct_test / float(N_test)
        print (f'TEST:   kappa = {self.model.finalized_kappa.item() :.5f}     | test loss= {loss.item():.3f}, test acc= {acc.item():.2f}% |')
        test_acc = (acc.item())
        test_loss = (loss.item())

        return test_loss, test_acc
