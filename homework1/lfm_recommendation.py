import torch
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt



## set randn seed
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


## mean square error
def mean_square_error(P, Q, bu, bi, mu, val_list):
    # R = P * Q^T
    error = 0
    for u, i, r in val_list:
        error += ((r - (mu+bu[u]+bi[i]+torch.dot(P[u], Q[i])))**2).item() #loss += (r -(mu + bu[u] + bi[i] + torch.dot(P[u], Q[i])))**2
    # error = error**0.5
    return error/len(val_list)


def draw_loss(train_mean_square_loss, train_l2_loss, val_loss, fig_name="loss.png"):
    
    plt.plot(train_l2_loss, label='train loss l2')
    plt.plot(train_mean_square_loss, label='train loss mean square')
    plt.plot([i+j for i, j in zip(train_l2_loss, train_mean_square_loss)], label='train loss')
    plt.plot(val_loss, label='val loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    # plt.show()
    plt.savefig(fig_name)
    plt.close()


def read_data():
    ## read the user-item matrix from col_matirx.csv using pandas
    data = pd.read_csv('col_matrix.csv', header=None)
    print("data shape: " ,data.shape)
    print("data sample: ", data.head())
    print("data values: ", data.values)
    ## the data need to svd 
    R = data.values
    ## split the data into training, validation and testing set
    ## exlude all zero values
    ## we use matrix[2000:4100, 2700:] as the validation set and other non-zero values as the training set
    train_list = []
    val_list = []
    val_u_start = 2000
    val_u_end = 4100
    val_i_start = 2700
    val_i_end = R.shape[1]
    average = 0
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if R[i,j] != 0:
                if i >= val_u_start and i<val_u_end and j >= val_i_start and j<val_i_end:
                    val_list.append((i,j,R[i,j]))
                else:
                    train_list.append((i,j,R[i,j]))
                    average += R[i,j]
    average = average/len(train_list)

    print("train list len: ", len(train_list), " val list len: ", len(val_list), "average rating: ", average)
    print(train_list[:10])
    print(val_list[:10])
    return R, train_list, val_list, average


def save_test_results(P, Q, save_path, mu=None, bu=None, bi=None, val_list=None):

    test_start_u = 4100
    test_start_i = 2700


    R = torch.mm(P, Q.t())

    if mu is not None:
        R = R+bu.view(-1,1)+bi.view(1,-1)+mu

    if val_list is not None:   
        error = 0
        for u, i, r in val_list:
            error += ((r - R[u,i])**2).item()
        error = error/len(val_list)
        print("validation mean square error: ", error**0.5)
    
    save_R = R[test_start_u:, test_start_i:]
    ##covert to int and range to 0 and 5

    save_R = torch.clamp(save_R, 0, 5)
    save_R = torch.round(save_R)
    save_R = save_R.to(dtype=torch.int32)

    ## convert to numpy array
    save_R = save_R.cpu().detach().numpy()
    print("save_R shape: ", save_R.shape)
    ## save the results to csv file
    np.savetxt(save_path, save_R, delimiter=",")



        
# don't use the  regularization
def train_no_regularization():
    print("train_no_regularization")
    R, train_list, val_list, average = read_data()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ## initialize P and Q, R = P * Q^T
    K = args.K
    P = torch.randn(R.shape[0], K, requires_grad=True, device=device)
    Q = torch.randn(R.shape[1], K, requires_grad=True, device=device)
    ## add bias 
    # bu = torch.randn(R.shape[0], requires_grad=True, device=device)
    # bi = torch.randn(R.shape[1], requires_grad=True, device=device)
    # mu = average
    # torch.nn.init.kaiming_uniform_(P, a=0, mode='fan_in', nonlinearity='leaky_relu')

    lr, num_epochs = args.lr, args.num_epochs
    l2_lamda1 = args.l2_lamda1
    l2_lamda2 = args.l2_lamda2
    batch_size = args.batch_size
    optimizer = torch.optim.Adam([P, Q], lr=lr) if args.optim == "adam" else torch.optim.SGD([P, Q], lr=lr, momentum=0.9)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.25)
    # train_l2_loss = []
    train_mean_square_loss = []
    val_loss = []
    best_val_loss = float('inf')
    for epoch in range(num_epochs):

        mean_losses = []
        # l2_losses = [] 
        for i in tqdm(range(0, len(train_list), batch_size)):
            batch = train_list[i:i+batch_size]
            loss = 0 
            # l2_reg = 0
            for u, i, r in batch:
                loss += (r -torch.dot(P[u], Q[i]))**2
                # l2_reg += l2_lamda1*(torch.norm(P[u], 2)**2) + l2_lamda2*(torch.norm(Q[i], 2)**2) + l2_lamda1*(bu[u]**2) + l2_lamda2*(bi[i]**2)
            
            mean_losses.append(loss.item()/len(batch) )
            # l2_losses.append(l2_reg.item()/len(batch) )

            loss = loss/len(batch) #+ l2_reg/len(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print("loss: ", loss/len(batch), " l2: ", l2_reg/len(batch))
        #scheduler.step()    
        # val_temp_loss = mean_square_error(P, Q, bu, bi, mu, val_list)
        # train_l2_loss.append(sum(l2_losses)/len(l2_losses))

        val_temp_loss = 0
        for u, i, r in val_list:
            val_temp_loss += ((r - (torch.dot(P[u], Q[i])))**2).item()
        val_temp_loss = val_temp_loss/len(val_list)
        train_mean_square_loss.append(sum(mean_losses)/len(mean_losses))
        val_loss.append(val_temp_loss)
        print("epoch: ", epoch, " mean square: ", sum(mean_losses)/len(mean_losses) , " val loss: ", val_temp_loss)

        if val_temp_loss <= best_val_loss:
            print("save model at epoch: ", epoch)
            best_val_loss = val_temp_loss
            torch.save(P, f"K{K}_lr{lr}_l2{l2_lamda1}{l2_lamda2}_bs{batch_size}_no_regularization_P.pt")
            torch.save(Q, f"K{K}_lr{lr}_l2{l2_lamda1}{l2_lamda2}_bs{batch_size}_no_regularization_Q.pt")
            # torch.save(bu, f"K{K}_lr{lr}_l2{l2_lamda1}{l2_lamda2}_bs{batch_size}_bu.pt")
            # torch.save(bi, f"K{K}_lr{lr}_l2{l2_lamda1}{l2_lamda2}_bs{batch_size}_bi.pt")
    print(train_mean_square_loss, "\n" ,val_loss)
    

    plt.plot(train_mean_square_loss, label='train loss mean square')
  
    plt.plot(val_loss, label='val loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    # plt.show()
    plt.savefig(f"K{K}_lr{lr}_l2{l2_lamda1}{l2_lamda2}_bs{batch_size}_optim{args.optim}_no_regularization.png")
    plt.close()

    #save results
    save_test_results(P, Q, f"K{K}_lr{lr}_l2{l2_lamda1}{l2_lamda2}_bs{batch_size}_no_regularization_results.csv", val_list=val_list)

# don't use the bias
def train_no_bias():
    print("train_no_bias")
    R, train_list, val_list, average = read_data()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ## initialize P and Q, R = P * Q^T
    K = args.K
    P = torch.randn(R.shape[0], K, requires_grad=True, device=device)
    Q = torch.randn(R.shape[1], K, requires_grad=True, device=device)
    # ## add bias 
    # bu = torch.randn(R.shape[0], requires_grad=True, device=device)
    # bi = torch.randn(R.shape[1], requires_grad=True, device=device)
    # mu = average
    # torch.nn.init.kaiming_uniform_(P, a=0, mode='fan_in', nonlinearity='leaky_relu')

    lr, num_epochs = args.lr, args.num_epochs
    l2_lamda1 = args.l2_lamda1
    l2_lamda2 = args.l2_lamda2
    batch_size = args.batch_size
    optimizer = torch.optim.Adam([P, Q], lr=lr) if args.optim == "adam" else torch.optim.SGD([P, Q], lr=lr, momentum=0.9)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.25)
    train_l2_loss = []
    train_mean_square_loss = []
    val_loss = []
    best_val_loss = float('inf')
    for epoch in range(num_epochs):

        mean_losses = []
        l2_losses = [] 
        for i in tqdm(range(0, len(train_list), batch_size)):
            batch = train_list[i:i+batch_size]
            loss = 0 
            l2_reg = 0
            for u, i, r in batch:
                loss += (r -torch.dot(P[u], Q[i]))**2
                l2_reg += l2_lamda1*(torch.norm(P[u], 2)**2) + l2_lamda2*(torch.norm(Q[i], 2)**2)
            
            mean_losses.append(loss.item()/len(batch) )
            l2_losses.append(l2_reg.item()/len(batch) )

            loss = loss/len(batch) + l2_reg/len(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print("loss: ", loss/len(batch), " l2: ", l2_reg/len(batch))
        #scheduler.step()    
        # val_temp_loss = mean_square_error(P, Q, bu, bi, mu, val_list)
        val_temp_loss = 0
        for u, i, r in val_list:
            val_temp_loss += ((r - (torch.dot(P[u], Q[i])))**2).item()
        val_temp_loss = val_temp_loss/len(val_list)
        train_l2_loss.append(sum(l2_losses)/len(l2_losses))
        train_mean_square_loss.append(sum(mean_losses)/len(mean_losses))
        val_loss.append(val_temp_loss)
        print("epoch: ", epoch, " train loss, l2: ",sum(l2_losses)/len(l2_losses), " mean square: ", sum(mean_losses)/len(mean_losses) , " val loss: ", val_temp_loss)

        if val_temp_loss <= best_val_loss:
            print("save model at epoch: ", epoch)
            best_val_loss = val_temp_loss
            torch.save(P, f"K{K}_lr{lr}_l2{l2_lamda1}{l2_lamda2}_bs{batch_size}_no_bias_P.pt")
            torch.save(Q, f"K{K}_lr{lr}_l2{l2_lamda1}{l2_lamda2}_bs{batch_size}_no_bias_Q.pt")
            # torch.save(bu, f"K{K}_lr{lr}_l2{l2_lamda1}{l2_lamda2}_bs{batch_size}_bu.pt")
            # torch.save(bi, f"K{K}_lr{lr}_l2{l2_lamda1}{l2_lamda2}_bs{batch_size}_bi.pt")
    print(train_mean_square_loss, "\n", train_l2_loss, "\n" ,val_loss)
    draw_loss(train_mean_square_loss, train_l2_loss, val_loss, fig_name=f"K{K}_lr{lr}_l2{l2_lamda1}{l2_lamda2}_bs{batch_size}_optim{args.optim}_no_bias.png")

    #save results
    save_test_results(P, Q, f"K{K}_lr{lr}_l2{l2_lamda1}{l2_lamda2}_bs{batch_size}_no_bias_results.csv", val_list=val_list)


def train():
    print("bias + regularization")
    R, train_list, val_list, average = read_data()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ## initialize P and Q, R = P * Q^T
    K = args.K
    P = torch.randn(R.shape[0], K, requires_grad=True, device=device)
    Q = torch.randn(R.shape[1], K, requires_grad=True, device=device)
    ## add bias 
    bu = torch.randn(R.shape[0], requires_grad=True, device=device)
    bi = torch.randn(R.shape[1], requires_grad=True, device=device)
    mu = average
    # torch.nn.init.kaiming_uniform_(P, a=0, mode='fan_in', nonlinearity='leaky_relu')

    lr, num_epochs = args.lr, args.num_epochs
    l2_lamda1 = args.l2_lamda1
    l2_lamda2 = args.l2_lamda2
    batch_size = args.batch_size
    optimizer = torch.optim.Adam([P, Q], lr=lr) if args.optim == "adam" else torch.optim.SGD([P, Q], lr=lr, momentum=0.9)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.25)
    train_l2_loss = []
    train_mean_square_loss = []
    val_loss = []
    best_val_loss = float('inf')
    for epoch in range(num_epochs):

        mean_losses = []
        l2_losses = [] 
        for i in tqdm(range(0, len(train_list), batch_size)):
            batch = train_list[i:i+batch_size]
            loss = 0 
            l2_reg = 0
            for u, i, r in batch:
                loss += (r -(mu + bu[u] + bi[i] + torch.dot(P[u], Q[i])))**2
                l2_reg += l2_lamda1*(torch.norm(P[u], 2)**2) + l2_lamda2*(torch.norm(Q[i], 2)**2) + l2_lamda1*(bu[u]**2) + l2_lamda2*(bi[i]**2)
            
            mean_losses.append(loss.item()/len(batch) )
            l2_losses.append(l2_reg.item()/len(batch) )

            loss = loss/len(batch) + l2_reg/len(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print("loss: ", loss/len(batch), " l2: ", l2_reg/len(batch))
        #scheduler.step()    
        val_temp_loss = mean_square_error(P, Q, bu, bi, mu, val_list)
        train_l2_loss.append(sum(l2_losses)/len(l2_losses))
        train_mean_square_loss.append(sum(mean_losses)/len(mean_losses))
        val_loss.append(val_temp_loss)
        print("epoch: ", epoch, " train loss, l2: ",sum(l2_losses)/len(l2_losses), " mean square: ", sum(mean_losses)/len(mean_losses) , " val loss: ", val_temp_loss)

        if val_temp_loss <= best_val_loss:
            print("save model at epoch: ", epoch)
            best_val_loss = val_temp_loss
            torch.save(P, f"K{K}_lr{lr}_l2{l2_lamda1}{l2_lamda2}_bs{batch_size}_P.pt")
            torch.save(Q, f"K{K}_lr{lr}_l2{l2_lamda1}{l2_lamda2}_bs{batch_size}_Q.pt")
            torch.save(bu, f"K{K}_lr{lr}_l2{l2_lamda1}{l2_lamda2}_bs{batch_size}_bu.pt")
            torch.save(bi, f"K{K}_lr{lr}_l2{l2_lamda1}{l2_lamda2}_bs{batch_size}_bi.pt")
    print(train_mean_square_loss, "\n", train_l2_loss, "\n" ,val_loss)
    draw_loss(train_mean_square_loss, train_l2_loss, val_loss, fig_name=f"K{K}_lr{lr}_l2{l2_lamda1}{l2_lamda2}_bs{batch_size}_optim{args.optim}.png")
    ## save results
    save_test_results(P, Q, f"K{K}_lr{lr}_l2{l2_lamda1}{l2_lamda2}_bs{batch_size}_results.csv", mu=mu, bu=bu, bi=bi, val_list=val_list)



if __name__ == "__main__":   

  




    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--K', type=int, default=30, help='number of latent factors')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--l2_lamda1', type=float, default=0.04, help='l2 regularization weight for P')
    parser.add_argument('--l2_lamda2', type=float, default=0.04, help='l2 regularization weight for Q')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=20, help='number of epochs')
    parser.add_argument("--optim", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--action", type=int, default=0, choices=[0, 1, 2], help="0: bias + regularization, 1: no bias, 2: no regularization")
    args = parser.parse_args()


    setup_seed(42)

    if args.action == 0:
        train() #bias + regularization
    elif args.action == 1:
        train_no_bias() # no bias
    elif args.action == 2:
        train_no_regularization() # no regularization






    
    