# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, freqz, firwin2

plt.close('all')

d_n = np.zeros(10000) # noisy ECG data
d_n = np.genfromtxt("data.txt")

#####################################################################################
def fir_lowpass_filter(numtaps,freq_array,gain_array,data):
    b = firwin2(numtaps,freq_array,gain_array) # linear phase FIR filter to avoid distortion
    w, h = freqz(b, [1.0], worN=8192)        
    y = lfilter(b, [1.0], data) # denominator polynomial is just 1
    return y

def fir_highpass_filter(numtaps,freq_array,gain_array,data):
    b = firwin2(numtaps,freq_array,gain_array,antisymmetric='True') # linear phase FIR filter to avoid distortion
    w, h = freqz(b, [1.0], worN=8192)        
    y = lfilter(b, [1.0], data) # denominator polynomial is just 1
    return y

def fir_lowpass_filter_plot(numtaps,freq_array,gain_array,fs):
    b = firwin2(numtaps,freq_array,gain_array) # linear phase FIR filter to avoid distortion
    w, h = freqz(b, [1.0], worN=8192)        
    
    plt.figure()
    plt.plot(b, 'bo-', linewidth=2)
    plt.title('Lowpass filter Coefficients (%d taps)' % numtaps)
    plt.grid(True)
    
    plt.figure()
    plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
    plt.xlim(0, 0.5*fs)
    plt.title("Lowpass Filter Frequency Response")
    plt.xlabel('Frequency [Hz]')
    plt.grid()
    return

def fir_highpass_filter_plot(numtaps,freq_array,gain_array,fs):
    b = firwin2(numtaps,freq_array,gain_array,antisymmetric='True') # linear phase FIR filter to avoid distortion
    w, h = freqz(b, [1.0], worN=8192)        
    
    plt.figure()
    plt.plot(b, 'bo-', linewidth=2)
    plt.title('Highpass filter Coefficients (%d taps)' % numtaps)
    plt.grid(True)
    
    plt.figure()
    plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
    plt.xlim(0, 0.5*fs)
    plt.title("Highpass Filter Frequency Response")
    plt.xlabel('Frequency [Hz]')
    plt.grid()
    return

def lowpass_highpass(data, freq_array_lp, gain_array_lp, freq_array_hp, gain_array_hp, numtaps):
    # Lowpass
    data_lowpassed = fir_lowpass_filter(numtaps,freq_array_lp,gain_array_lp,data)
    data_lowpassed = np.delete(data_lowpassed,np.arange(numtaps-1))    # discard initial corrupted samples
    
    # Highpass
    data_lowpassed_highpassed = fir_highpass_filter(numtaps,freq_array_hp,gain_array_hp,data_lowpassed)
    data_lowpassed_highpassed = np.delete(data_lowpassed_highpassed,np.arange(numtaps-1))    # discard initial corrupted samples
    
    return data_lowpassed_highpassed, data_lowpassed

def find_start_points(data):
    array1 = np.zeros(len(data[0:-1]))
    array1[np.where(data[0:-1]==0)] = 1        
    array2 = np.zeros(len(data[1:]))
    array2[np.where(data[1:]>0)] = 1
    start_points = np.multiply(array1,array2)        
    return start_points

def find_end_points(data):
    array1 = np.zeros(len(data[1:]))
    array1[np.where(data[1:]==0)] = 1
    array2 = np.zeros(len(data[0:-1]))
    array2[np.where(data[0:-1]>0)] = 1
    end_points = np.multiply(array1,array2)
    return end_points

def remove_isolated_jumps(start_indices,end_indices):
    start_indices_1 = np.copy(start_indices)
    end_indices_1 = np.copy(end_indices)
    for i in range(len(start_indices)):
        if i==0:
            if (start_indices[i+1]-start_indices[i])>25: 
                start_indices_1[i] = -1 #isolated peak; remove
                end_indices_1[i] = -1
        elif i==len(start_indices)-1:
            if (start_indices[i]-start_indices[i-1])>25:
                start_indices_1[i] = -1 #isolated peak; remove
                end_indices_1[i] = -1
        else:
            if (start_indices[i]-start_indices[i-1])>25 and (start_indices[i+1]-start_indices[i])>25:
                start_indices_1[i] = -1 #isolated peak; remove
                end_indices_1[i] = -1    
    start_indices_1 = np.delete(start_indices_1,np.where(start_indices_1==-1)[0])
    end_indices_1 = np.delete(end_indices_1,np.where(end_indices_1==-1)[0])
    return start_indices_1, end_indices_1


def merge_closeby_windows(start_indices_1,end_indices_1):
    start_indices_2 = np.copy(start_indices_1)
    end_indices_2 = np.copy(end_indices_1)
    for i in range(len(start_indices_1)):    
        if i==len(start_indices_1)-1:
            pass
        elif (start_indices_1[i+1]-start_indices_1[i])<25:
            start_indices_2[i+1] = -1 # merge closeby peaks to create window
            end_indices_2[i] = -1
    start_indices_2 = np.delete(start_indices_2,np.where(start_indices_2==-1)[0])
    end_indices_2 = np.delete(end_indices_2,np.where(end_indices_2==-1)[0])
    return start_indices_2, end_indices_2


def clean_edge_windows(start_indices,end_indices):
    if abs(len(start_indices)-len(end_indices))>1:
        print("ERROR. Check start and end indices")
    elif (len(start_indices)-len(end_indices))==1:
        end_indices = np.append(end_indices,len(d_diff))
    elif (len(start_indices)-len(end_indices))==-1:
        start_indices = np.append(0,start_indices)
    return start_indices, end_indices

######## Filter the input data 
fs = 500.0  # sample rate, Hz
numtaps = 80

# Lowpass filter coefficients and response
freq_array_lp = np.array([0, 20, 25, (fs/2.0)])/(fs/2.0) # remove noise higher than 1 Hz
gain_array_lp = np.array([1, 0.5, 0, 0])
fir_lowpass_filter_plot(numtaps,freq_array_lp,gain_array_lp,fs)

# Highpass filter coefficients and response
freq_array_hp = np.array([0, 1, 2, (fs/2.0)])/(fs/2.0)
gain_array_hp = np.array([0, 0.5, 1, 1])
fir_highpass_filter_plot(numtaps,freq_array_hp,gain_array_hp,fs)

# Apply lowpass followed by highpass filters
d_n_lphp,d_n_lp = lowpass_highpass(d_n, freq_array_lp, gain_array_lp, freq_array_hp, gain_array_hp, numtaps)

plt.figure()
plt.plot(d_n)
plt.title("Noisy ECG")
plt.show()  

plt.figure()
plt.plot(d_n_lp)
plt.title("Noisy ECG post lowpass filtering")
plt.show()        

plt.figure()
plt.plot(d_n_lphp)
plt.title("Noisy ECG post lowpass, highpass filtering")
plt.show() 

##### Calculate derivative of the filtered data
d = np.copy(d_n_lphp) # short variable name "d" for convenience

d_diff = d[1:] - d[0:-1]
d_diff_abs = np.abs(d_diff)

plt.figure()
plt.plot(d_diff_abs)
plt.title("ECG absolute derivative post filtering")
plt.show()


####### Thresholding the derivative
d_diff_abs_thresholded = np.copy(d_diff_abs)
threshold = np.max(d_diff_abs_thresholded)*0.16
d_diff_abs_thresholded[np.where(d_diff_abs<threshold)]=0
plt.figure()
plt.plot(d_diff_abs_thresholded)
plt.title("Thresholded ECG absolute derivative post filtering")
plt.show()


####### Finding window start and end indices
start_points = find_start_points(d_diff_abs_thresholded)
end_points = find_end_points(d_diff_abs_thresholded)
start_indices = np.where(start_points==1)[0]
end_indices = np.where(end_points==1)[0]

####### Take care of edge windows
start_indices, end_indices = clean_edge_windows(start_indices,end_indices)

####### Remove isolated jumps (valid absolute derivative peaks come in groups of 2 or more in thresholded |dy1/dt|)
start_indices_1, end_indices_1 = remove_isolated_jumps(start_indices,end_indices)
start_indices_1, end_indices_1 = clean_edge_windows(start_indices_1,end_indices_1)

####### merge closeby windows
start_indices_2, end_indices_2 = merge_closeby_windows(start_indices_1,end_indices_1)
start_indices_2, end_indices_2 = clean_edge_windows(start_indices_2,end_indices_2)

####### Add 1 to the indices to appropriately apply on d instead of d_diff
start_indices_3 = start_indices_2 + 1
end_indices_3 = end_indices_2 + 1

####### Calculate peaks of the waveform in each window
peak_indices = np.zeros(len(start_indices_3))
for i in range(len(start_indices_3)):
    peak_indices[i] = start_indices_3[i] + np.argmax(d[start_indices_3[i]:end_indices_3[i]])

plt.figure()
plt.plot(d_n_lphp)
for i in range(len(peak_indices)):
    plt.axvline(x=peak_indices[i],color='r',ls='--')
plt.title("ECG post filtering with detected peaks")
plt.show()