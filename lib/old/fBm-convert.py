import numpy as np
import glob
import os

#path=/Users/wiese/tex/persistence/2absorbingboundaries/simulation/
path="/Users/wiese/fBm-simulations/"

allfiles=glob.glob(path+"*.npy")

for filename in allfiles:
    convertedfilename=filename.replace(".npy",".dat")
    print "convert ", filename, "to .dat"
    if os.path.getsize(filename) == 0 :
        print "WARNING : file is empty"
    else :    
        data = np.load(filename)
        openedfile=open(convertedfilename,"w")
        if len(data.shape) == 1 :
            for item in data:
                print>>openedfile, item
            openedfile.close
        else : # we suppose dimension two
            #print "ahllo", data.shape[1]
            for i in range(0,data.shape[1]-1) :
               # for j in (1,data.shape[2]) :
               print>>openedfile, str(data[i].tolist())[1:-1]  


#string= path+"fBm_X0_Histo(H=%s,T=%s,seed=%s)" % (H,N,Rseed)
#print "convert ", string+".npy to .dat"
#histoX0=np.load(string+".npy")
#file = open(string+".dat","w")
#for item in histoX0:
#  print>>file, item
#file.close

#string=path+"fBm_Tabsorb_Histo(H=%s,T=%s,seed=%s)" % (H,N,Rseed)
#print "convert ", string+".npy to .dat"
#histoTabsorb=np.load(string+".npy")
#file = open(string+".dat","w")
#for item in histoTabsorb:
#  print>>file, item
#file.close

