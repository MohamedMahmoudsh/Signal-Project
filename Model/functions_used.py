def find_the_best_threshold(model , X, Y ):
    """
    To find the best threshold I need the model , X , Y to selct the best threshold passed on acciracy
    The function return a list of tuble the sample of tuple (accuracy at threshold , threshold )
   """
    threshold_VS_accuracy = []
    Thresholds = np.linspace(0,1 ,100)
    Predection = model(X).numpy()
    for Threshold in Thresholds:
        Y_hat = Predection >= Threshold
        Y_hat = Y_hat.reshape((-1,))
        True_rate =  Y_hat == Y
        threshold_VS_accuracy.append((Threshold ,  True_rate.sum()))
    return threshold_VS_accuracy
  


def evaluate(Predection, X ,Y, threshold):
    """
    This function evaluate True postive rate  , False postive rate at a certain threshold.
    This function is used to evaluate the ROC curve.
    This function return (FP_rate , TP_rate)
    """
    condition = Predection >=threshold
    Y_hat =condition.reshape((-1,))
    TP = Y_hat[Y==1].sum()
    TP_rate = TP / len(Y_hat[Y==1])
    FP = Y_hat[Y==0].sum()
    FP_rate = FP/len(Y_hat[Y==0])
    #return FP , TP          #Test Condition
    return FP_rate , TP_rate
  

def ROC_Pair_FN(model ,X , Y):
    """
    This function is uesed in ploting the ROC curve
    This function needs model , X , Y
    output (FP_rate , TP_rate )
    """
    Predection  = model(X).numpy()
    Thresholds = np.linspace(0,1 ,100)
    ROC_pair = []
    for Threshold in Thresholds:
        pair = evaluate(Predection , X , Y , Threshold)
        ROC_pair.append(pair)
    return np.c_[np.array(ROC_pair) , Thresholds]


def AUC_For_model(ROC_Pair):
    """
    This Function is used to evaluate the area under ROC curve which used to evaluate the model
    bassed on ROC curve.
    return AUC
    """
    AUC = 0
    for i in range(1 , len(ROC_Pair)):
        AUC += abs(ROC_Pair[i,0] - ROC_Pair[i-1,0] ) * (ROC_Pair[i,1] + ROC_Pair[i-1,1] )
    return AUC/2



def read_signal(directory):
  """
  The function is used to read signal from its directory
  The function takes the directory as input 
  The function output is a signal in form of numpy array
  
  """
    signal  = wfdb.rdrecord(directory,sampfrom=0 , sampto=5000)
    signal = signal.__dict__['p_signal'][::,0]
    return signal


def preprocess(dat):
  """
  This function read the data from directory and store it with label in numpy array
  Input DataFrame contain directory and label as columns
  Output two numpy array one for X , and one for y
  """
    data_dir = list(dat['directory'])
    data_signal = map(read_signal , data_dir)
    data_signal = list(data_signal)
    data_signal = np.array(data_signal)
    data_dict = {'SR' : 0 , 'SB': 1  }
    encoded_label = dat['diagnosis'].map(data_dict)
    return np.array(data_signal)  , np.array(encoded_label)



