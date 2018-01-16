import pyedflib
import numpy as np
import matplotlib.pyplot as plt
import random
import csv
import glob

###support functions#####################################

def pickstages(pieces, epochtime, timelist, stagelist,pickdic):
    pickeddic={}
    xindex=[]
    ylabel=[]
    for i in range(pieces):

        x = random.randint(0,1)
        if x == 1:
            time = i*30
            sleepstage = getlabel(time, timelist, stagelist)
            #print(sleepstage)
            pickeddic=addpicknum(pickeddic,sleepstage)

            if sleepstage in picklabel and pickeddic[sleepstage] <= pickdic[sleepstage]:
                xindex.append(i)
                ylabel.append(sleepstage)

    return xindex,ylabel
                
            
    
def addpicknum(alreadypic,sleepstage):
    alreadypic[sleepstage] = alreadypic.get(sleepstage,0)+1
    return alreadypic
    


def getlabel(time,timelist,stagelist):
    i=0
    limit = len(timelist)-2
    while time > timelist[i]:
        if i <= limit:
            i += 1
        else:
            break
    return stagelist[i-1]
    

def fourierfunc(Fs, signal):
    y=signal
    n = int(len(y))
    k = np.arange(n)
    T = n/Fs
    frq = k/T
    frq=frq[range(divmod(n,2)[0])]
    Y = np.fft.fft(y)/n
    Y = Y[range(divmod(n,2)[0])]
    index=[]
    frqnew=[]
    
    num=0
    for i in frq:
        if i >=  0.5 and i <= 35:
            index.append(num)
            frqnew.append(i)
        num += 1

    Y = Y[index]

    return abs(Y),frqnew


###load data #######################################


labelname = []

for filename in glob.glob('*-Hypnogram.edf'):
    labelname.append(filename)


wavename = []

for filename in glob.glob('*-PSG.edf'):
    wavename.append(filename)



####number limit########################################################################################################################################
# parameters

Fs = 100
epochtime = 30


picklabel=['Sleep stage W', 'Sleep stage 1', 'Sleep stage 2', 'Sleep stage 3','Sleep stage 4','Sleep stage R']


pickdic = {'Sleep stage W':253, 'Sleep stage 1':10, 'Sleep stage 2':61, 'Sleep stage 3':11, 'Sleep stage 4': 8, 'Sleep stage R':26}

##maxlimit = 10000000000
##pickdic = {'Sleep stage W':maxlimit, 'Sleep stage 1':maxlimit, 'Sleep stage 2':maxlimit, 'Sleep stage 3':maxlimit, 'Sleep stage 4': maxlimit, 'Sleep stage R':maxlimit}

wavenumberlimit = 40

####filename#################################################################################################################################################


numstring = "1"


traindatafilename = 'train'+numstring+'.csv'
trainlabeldatafilename = 'label_train'+numstring+'.csv'


testdatafilename = 'test'+numstring+'.csv'
testlabeldatafilename = 'label_test'+numstring+'.csv'




###training data#############################################################################################################################################

print("[train data begins]")

filenum = 0


wavenumbercount = 0



for wn in wavename:

    if wavenumbercount >wavenumberlimit:
        continue



  
    wavefilename = wn
    #print(wavefilename)
    for ln in labelname:
        if ln[2:6] == wn[2:6]:
            labelfilename = ln
        else:
            continue

        ### read data and label files
        f = pyedflib.EdfReader(wavefilename)
        n = f.signals_in_file

        labelf = pyedflib.EdfReader(labelfilename)
        annotations = labelf.readAnnotations()


        timelist = annotations[0]
        stagelist = annotations[2]

        #print(timelist)


        i=0 # EEG 2 

        signal_labels = f.getSignalLabels()

        sigbufs = np.zeros((n, f.getNSamples()[i]))

        signalnum=f.getNSamples()[i]


        #print(signalnum)

        sigbufs[0] = f.readSignal(i)

        #print(sigbufs[0][1000:1200])


        pieces=divmod(signalnum,epochtime*Fs)[0]

        #print(pieces)

        x,y=pickstages(pieces, epochtime, timelist, stagelist,pickdic)

        #print(y)


        count =0



        if filenum == 0:
            
            with open(traindatafilename, 'w', newline='') as combineFile:
                abcsv = csv.writer(combineFile, dialect='excel')

                with open(trainlabeldatafilename, 'w', newline='') as combineFile:
                    abcsvlabel = csv.writer(combineFile, dialect='excel')


                    for i in x:
                        signal = sigbufs[0][i*100:(i+30)*100]
                        Y,frq = fourierfunc(Fs, signal)
            

                        abcsv.writerow(Y)

                        label=[]
                        label.append(y[count])

                        

                        abcsvlabel.writerow(label)

##                        if count == 1:
##                            plt.figure(1)
##                            plt.plot(frq,Y,'b')
##                            plt.ylabel('Amplitude [au]', fontsize=14)
##                            plt.xlabel('Frequency [Hz]', fontsize=14)
##                            plt.axis([0, 40, 0, 10])
##                            plt.savefig('FFT.eps', format='eps')
##                            plt.grid(color='grey', linestyle='--', linewidth=1)
##                            #plt.savefig("FFT.png", dpi=300)
##
##                            
##                            plt.show()

                        count += 1

            print('wave '+str(filenum)+' has finished')
            filenum += 1
        else:
            with open(traindatafilename, 'a', newline='') as combineFile:
                abcsv = csv.writer(combineFile, dialect='excel')

                with open(trainlabeldatafilename, 'a', newline='') as combineFile:
                    abcsvlabel = csv.writer(combineFile, dialect='excel')


                    for i in x:
                        signal = sigbufs[0][i*100:(i+30)*100]
                        Y,frq = fourierfunc(Fs, signal)
            

                        abcsv.writerow(Y)

                        label=[]
                        label.append(y[count])

                        abcsvlabel.writerow(label)

                        count += 1
            print('wave '+str(filenum)+' has finished')
            filenum += 1


    wavenumbercount += 1

############################################################################################################################################################






