import numpy as np
import numpy.random as rand
import sys
import os
import socket
import time

### PARAMETERS ###
N = 2**16 #Size of the lattice 2**12 for Mathieu,  2**13 for Kay+M+Tridib F2 for postime, Tmax and Tlast
isave=10000
ech = 10**8
#Numbers of numerical runs; at least 10^9, 1*10**10 was in the above code
H=0.6  #Hurst exponent
Rseed=10000
if len(sys.argv)>1:
    print sys.argv
    Rseed=int(sys.argv[1]);
    
xresolution=1000   # 10000
Tresolution=1000   # 10000
if H==0.33 :
    mult=1.7  # mult=1.7 for H=0.33,  failures =  382 / 10000000
elif H==0.4 :    
    mult=1.25 
elif H==0.45 :    
    mult=1.05 
elif H==0.475 :    
    mult=1.0 
elif H==0.525 :    
    mult=0.8 
elif H==0.55 :    
    mult=0.7 
elif H==0.6 :    
    mult=0.6 
elif H==0.67 :    
    mult=0.5  # mult=0.5 for H=0.66
elif H==0.75 :    
    mult=0.35 # failures = 205 / 1000000 for N=2**13
else :
    mult=1    
threshold = mult
path="/users/wiese/fBm-simulations/"   #attention, on corto it's /users with miniscule "u"
starttime= os.times()
rand.seed(Rseed)

### MAIN ###

def gamma(k):
	return (abs(k-1)**(2.*H)+abs(k+1)**(2.*H)-2.*abs(k)**(2.*H))

rr=np.zeros(2*N)
for k in range(0,N):
	rr[k]=gamma(k)
for k in range(N+1,2*N):
	rr[k]=gamma(2*N-k)

ll=np.sqrt(np.real(np.fft.ifft(rr)))
del(rr)

Correl=((1.*np.arange(1,N+1)/N)**(2*H)+1.-(1.-1.*np.arange(1,N+1)/N)**(2*H))/2. # This is used only to generate bridges


W=np.zeros(2*N,complex)
#histoPos=np.zeros(N+1) # histogram for the temporal observables. Create more variables to save more observables
#histoLastZero=np.zeros(N) # histogram for the temporal observables. Create more variables to save more observables
#histoTmax=np.zeros(N) # histogram for the temporal observables. Create more variables to save more observables
histoX0=np.zeros(xresolution+1,np.int32)
histoTofX0=np.zeros(xresolution+1,np.int32)
histoX0END=np.zeros(xresolution+1,np.int32)
histoTabsorb=np.zeros(Tresolution+1)
failures=0

print "Starting simulation for N = 2^",np.log(N)/np.log(2),", H=", H, " with ",ech," samples."
print "threshold =",threshold
print "seed =", Rseed
print "process ID =", os.getpid()
print "path =", path
print "hostname =", socket.gethostname()
print "started:",time.strftime("%d.%m.%Y, at %H:%M:%S")
sys.stdout.flush()

i=0
while i<ech:  #Main loop
	### Generation of a random path
	V=rand.normal(0,1,2*N)

	W[0]=V[0]
	W[N]=V[N]
	W[1:N]=(V[1:N] + V[N+1:2*N]*1j)/np.sqrt(2.)
	temp=(V[1:N]-V[N+1:2*N]*1j)/np.sqrt(2.)
	W[N+1:2*N]=temp[::-1]
	
	Z=np.real(np.fft.fft(ll*W))
	
	X=np.cumsum(Z[0:N])/N**H # X is the fBm path define on [0,1]. Attention : X[0] is the process at time 1/N, and X[N-1] is the process at time 1.
	#B=X-(X[N-1])*Correl # This transform the path to a bridge, with again B[0] the process at time 1/N. Comment it if not needed.

	
	### The observables (change X to B for bridge observables)
	
	#tau=np.sum(X>0) # The positive time of the fBm path
	#histoPos[tau]+=1 # Storing the value of the positive time in an histogram. It is also possible to store all the values in a list, but it takes a lot more memory.
	
	
	#temp1=X[::-1]
	#temp2=(temp1[0:N-1]*temp1[1:N]<0)*1
	#if (np.sum(temp2)) < 1: # to avoid error when there is no last zero in the realization
	#   LastZero=0
	#else:
	#   LastZero=N-1-list(temp2).index(1) # The last zero of the fBm path
	#histoLastZero[LastZero]+=1
	
	#Tmax=list(X).index(np.max(X)) # The time of the max
	#histoTmax[Tmax]+=1
	#m=np.max(X) # and the max value
	
	# the point X0 above which it will go to the upper boundary, and below to the lower
	X2=np.maximum.accumulate(X) 
	X2=np.maximum(X2,0)   #our list of X is missing the first data-point 0
	X1=np.minimum.accumulate(X)
	X1=np.minimum(X1,0)   #our list of X is missing the first data-point 0
	
	#print X
	#print "X2=",X2
	#print "X1=",X1
	#print "X2-X1=",X2-X1
	
	Tabsorbed=np.searchsorted(X2-X1,threshold)+0  # this shift might be tricky   
	#print "Tabsorbed=",Tabsorbed, "/",N
	
	if Tabsorbed >= N:
	   #print "failure"
	   failures += 1
	   histoTabsorb[-1] += 1
	else:   
	   # linear interpolationh between the last two points 
	   deltaX2=X2[Tabsorbed]-X2[Tabsorbed-1]
	   deltaX1=X1[Tabsorbed]-X1[Tabsorbed-1]
	   y=(threshold-X2[Tabsorbed-1]+X1[Tabsorbed-1])/(deltaX2-deltaX1)     #fraction at which the threshold is attained
	   X1final=X1[Tabsorbed-1]+y*deltaX1
	   X2final=X2[Tabsorbed-1]+y*deltaX2
	   #print"y=",y
	   #print"deltaX2=",deltaX2
           #       print"deltaX1=",deltaX1
           #print"X2final=",X2final
           #print"X1final=",X1final
           #print"X2final-X1final=",X2final-X1final
	   X0=X2final/threshold        #everything bigger than X0 will get absorbed
	   X0index=int(np.rint(X0*xresolution))
	   #print "(X2-X1)[Tabsorbed]=",X2[Tabsorbed]-X1[Tabsorbed]
	   #print "(X2-X1)[Tabsorbed-1]=",X2[Tabsorbed-1]-X1[Tabsorbed-1]
	   #print "X2[Tabsorbed]=",X2[Tabsorbed]
	   #print "X1[Tabsorbed]=",X1[Tabsorbed]
	   #print "X0=",X0, ", index=",X0index
           histoX0[X0index]+=1
           histoX0END[int(np.rint(xresolution*X2[-1]/(X2[-1]-X1[-1])))]+=1
           histoTofX0[X0index]+=(Tabsorbed+y)/N

           histoTabsorb[int(np.rint((Tabsorbed+y)/N*(Tresolution)))]+=1
	#print "************"

	### Statistics
	
	if i%(isave)==0: # To run the "saving routine" only every isave paths
		#print (1.*i/ech*10**2), "%" , ": ",time.strftime("%d.%m.%Y, at %H:%M:%S")
		print socket.gethostname(), os.getpid(), ":", i, "iterations = ",(1.*i/ech*10**2), "%" , "after", os.times()[-1]-starttime[-1],"s"
                sys.stdout.flush()
		#np.save("/Users/wiese/tex/persistence/simulations/fBm_PositiveHisto(H=%s,T=%s,seed=%s)" % (H,N,Rseed),histoPos) # This creates a file with the histogram of an observable (here the positive time)
		#np.save("/Users/wiese/tex/persistence/simulations/fBm_LastZeroHisto(H=%s,T=%s,seed=%s)" % (H,N,Rseed),histoLastZero)
		#np.save("/Users/wiese/tex/persistence/simulations/fBm_TmaxHisto(H=%s,T=%s,seed=%s)" % (H,N,Rseed),histoTmax)
                np.save(path+"fBm_X0_Histo(H=%s,T=%s,seed=%s)" % (H,N,Rseed),histoX0)
                np.save(path+"fBm_TofX0_Histo2(H=%s,T=%s,seed=%s)" % (H,N,Rseed),histoTofX0)
                np.save(path+"fBm_X0_HistoEnd(H=%s,T=%s,seed=%s)" % (H,N,Rseed),histoX0END)
                np.save(path+"fBm_Tabsorb_Histo(H=%s,T=%s,seed=%s)" % (H,N,Rseed),histoTabsorb)
	i+=1

print "failures = ", failures, "/", ech

if ech>10000: 
    file = open(path+"fBm_X0_Histo_Final(H=%s,T=%s,seed=%s).dat" % (H,N,Rseed),"w")
    for item in histoX0:
        print>>file, item
    file.close

    file = open(path+"fBm_Tabsorb_Histo_Final(H=%s,T=%s,seed=%s).dat" % (H,N,Rseed),"w")
    for item in histoTabsorb:
        print>>file, item
    file.close

    file = open(path+"fBm_Tabsorb_HistoEnd_Final(H=%s,T=%s,seed=%s).dat" % (H,N,Rseed),"w")
    for item in histoX0END:
        print>>file, item
    file.close
