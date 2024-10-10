import numpy as np 
import itertools as it

class PVI:
    def __init__(self, data, y_true, model: callable, n_boots: int, alpha: float, column_set: list) -> None:
        self.X = data
        self.y_true = y_true
        self.model = model
        self.n_boots = n_boots
        self.alpha = alpha
        if column_set == None:
            try:
                self.column_set = self.X.columns
                self.X = self.X.to_numpy()
            except AttributeError:
                print('You either didn\'t provide column labels or your data isn\'t a DataFrame or both.')
    @staticmethod
    def bootstrap(y_true, y_pred, n_boots):
        rng = np.random.default_rng(seed=0)
        n = len(y_true)
        error = y_true-y_pred
        S_CI = np.zeros((n_boots,))
        for i in range(n_boots):
            indx = rng.choice(n, size=n, replace=True)
            S_CI[i] = np.mean(np.square(error[indx]))
        return S_CI
    def first_order_pvi(self):
        rows, columns = self.X.shape
        V_y = np.var(self.y_true)
        PI = dict()
        CI = dict()
        rolled_data = np.roll(self.X, rows//2, axis=0)
        for i in range(columns):
            d = rolled_data.copy()
            d[:,i] = self.X[:,i]
            y_pred = self.model(d)
            PI[self.column_set[i]] = 1 - np.mean(np.square(self.y_true-y_pred))/(2*V_y)
            
            S_CI = 1 - self.bootstrap(self.y_true, y_pred, self.n_boots)/(2*V_y)
            p0,p1 = np.quantile(S_CI, [self.alpha, 1-self.alpha])
            CI[self.column_set[i]] = {f'{str(self.alpha)}th':p0,f'{str(1 - self.alpha)}th':p1,'Quantile Differnece':p1-p0}
        return PI, CI
        
    def second_order_pvi(self):
        rows, columns = self.X.shape
        V_y = np.var(self.y_true)
        PIk = dict()
        PI, CI = self.first_order_pvi()
        CI = dict()
        rolled_data = np.roll(self.X, rows//2, axis=0)
        for i,j in it.combinations(range(columns), r=2):
            d = rolled_data.copy()
            d[:,[i,j]] = self.X[:,[i,j]]
            y_pred = self.model(d)
            PIk[(self.column_set[i],self.column_set[j])] = 1 - np.mean(np.square(self.y_true-y_pred))/(2*V_y) - PI[self.column_set[i]] - PI[self.column_set[j]]
            
            S_CI = 1 - self.bootstrap(self.y_true, y_pred, self.n_boots)/(2*V_y) - PI[self.column_set[i]] - PI[self.column_set[j]]
            p0,p1 = np.quantile(S_CI, [self.alpha, 1-self.alpha])
            CI[(self.column_set[i],self.column_set[j])] = {f'{str(self.alpha)}th':p0,f'{str(1 - self.alpha)}th':p1,'Quantile Differnece':p1-p0}
        return PIk, CI
    
    def total_order_pvi(self):
        rows, columns = self.X.shape
        V_y = np.var(self.y_true)
        PT = dict()
        CI = dict()
        for i in range(columns):
            d = self.X.copy()
            k = np.roll(self.X[:,i], rows//2, axis=0)
            d[:, i] = k
            y_pred = self.model(d)
            PT[self.column_set[i]] = np.mean(np.square(self.y_true-y_pred))/(2*V_y)
            
            S_CI = self.bootstrap(self.y_true, y_pred, self.n_boots)/(2*V_y)
            p0,p1 = np.quantile(S_CI, [self.alpha, 1-self.alpha])
            CI[self.column_set[i]] = {f'{str(self.alpha)}th':p0,f'{str(1 - self.alpha)}th':p1,'Quantile Differnece':p1-p0}
        return PT, CI