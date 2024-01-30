import torch

class Task_Dataset(Dataset):
    def __init__(self, X : sp.sparse.csr.csr_matrix, 
                       y : np.ndarray):
        self.X = X
        self.y = torch.from_numpy(y).float()
        assert self.X.shape[0] == self.y.shape[0]
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):    
        X = torch.from_numpy(self.X[idx].astype(np.int8).todense()).float().squeeze()
        y = self.y[idx]
        return X, y

movie_ds = Task_Dataset(movie_X_train, movie_y_train)
movie_dl = DataLoader(movie_ds, batch_size = 64, shuffle = True)

yelp_ds = Task_Dataset(yelp_X_train, yelp_y_train)
yelp_dl = DataLoader(yelp_ds, batch_size = 64, shuffle = True)


