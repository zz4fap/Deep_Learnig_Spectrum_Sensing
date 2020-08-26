import numpy as np
import pickle


#%%

filename = './pkl_data/coperative/128_co.pkl'

snrs=""
mods=""
lbl =""
Xd = pickle.load(open(filename,'rb'),encoding='latin')
snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
X = []  
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod,snr)])
        for i in range(Xd[(mod,snr)].shape[0]):  lbl.append((mod,snr))
X = np.vstack(X)

#     use QAM16 signal only
lbl = np.array(lbl) 
index = np.where(lbl=='QAM16')[0]   
X = X[index]
lbl = lbl[index]


#%%

maxlen = X.shape[-1]
nodes = X.shape[1]
#%%

SNR = []
for item in lbl:
    SNR.append(item[-1])
SNR = np.array(SNR,dtype='int16')

noise_vectors = []
for i in range(X.shape[0]*nodes):
    real = np.random.randn(maxlen) 
    imag = np.random.randn(maxlen)
    complex_noise_vector = real + 1j*imag
    energy = np.sum(np.abs(complex_noise_vector)**2)
    noise_vector = complex_noise_vector / (energy**0.5)
    real = np.real(noise_vector)
    imag = np.imag(noise_vector)
    noise_vectors.append([real,imag])
noise_vectors = np.array(noise_vectors)   
noise_vectors = np.reshape(noise_vectors,(int(noise_vectors.shape[0]/nodes),nodes,2,noise_vectors.shape[-1]))

# one-hot label, [1,0] with signal, [0,1] noise only
dataset = np.concatenate((X,noise_vectors),axis=0)
labelset = np.concatenate(([[1,0]]*len(X),[[0,1]]*len(noise_vectors)),axis=0)
labelset = np.array(labelset,dtype='int16')
# use snr -100 to represent noise only samples
SNR = np.concatenate((SNR,[-100]*len(noise_vectors)),axis=0) 

print(dataset.shape)
print(labelset.shape)
print(SNR.shape)
#%%
   
aux =  dataset[38000,:,:,:].reshape(2,2,128)

#%%
total_group = dataset.shape[0]
nodes = dataset.shape[1] 
total_num = total_group*nodes
   
snrs = np.linspace(-20,19,40)
snrs = np.array(snrs,dtype='int16')
snr_type = len(snrs)