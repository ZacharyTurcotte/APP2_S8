import numpy as np
import matplotlib.pyplot as plt

m = np.array([3,-1])
sigma = np.array([[1,0],[0,1]]).reshape(2,2)
dim = 2
N = [10,20,50,100,500]
nb_essai = 10
Sigma_list = []

fig = False
mean_list = []
mean_sig_list = []
for n in N:
    print(n)
    m_x_list = np.zeros((nb_essai, dim))
    Sigma_list = np.zeros((nb_essai, dim, dim))
    for i in range(nb_essai):

        X_k = np.random.multivariate_normal(m,sigma,n)
        m_x = 1/n*np.sum(X_k,axis=0)
        B = np.transpose(X_k - m_x)
        Sigma = np.cov(B)

        Sigma_list[i] = Sigma
        m_x_list[i] = m_x

        if fig:
            plt.figure(i)
            plt.hist(X_k)

    m_x_list = np.array(m_x_list)
    Sigma_list = np.array(Sigma_list)

    m_m_x = 1/nb_essai*np.sum(m_x_list,axis=0)
    m_Sigma = 1/nb_essai*np.sum(Sigma_list,axis=0)
    mean_list.append(m_m_x)
    mean_sig_list.append(m_Sigma)
x_axis = [10,20,50,100,500]
x_axis = [1,2,3,4,5]
plt.figure(1)
plt.plot(x_axis,mean_list)

mean_sig_list = np.array(mean_sig_list)
mean_sig_list = np.reshape(mean_sig_list,(5,4))

plt.figure(2)
plt.plot(x_axis,mean_sig_list[:,0])
plt.figure(3)
plt.plot(x_axis,mean_sig_list[:,1],)
plt.figure(4)
plt.plot(x_axis,mean_sig_list[:,2],)
plt.figure(5)
plt.plot(x_axis,mean_sig_list[:,3],)
plt.show()
print("DONE")