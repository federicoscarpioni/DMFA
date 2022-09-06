"""
Dynamic Multi-Frequency Analysis
Authors: Federico Scarpioni and Nicolò Pianta
Based on the work of Alberto Battistel and Fabio La Mantia
Date: September 2022

This modules contains a series of function to perform the Dynamic Multi-Frequency
Analysis on digitally sampled current and voltage signals. The main code is after
the function declarations.

"""
import numpy as np
import time
import math
from numpy.fft import fft, fftshift, fftfreq, ifft, ifftshift
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib.ticker as ticker


def cut_integer_periods(voltage, current, time, multisine_freq, sampling_rate):
    '''
    Reduce the signal to have an integer number of multisine periods. In the
    Fourier analysis this avoids spectral leakage.
    '''
    fmin = min(multisine_freq)
    multisine_periods = int(np.floor(voltage.size/((1/fmin)/sampling_rate)))
    lim = int((voltage.size-(multisine_periods/fmin/sampling_rate))/2)
    if lim != 0:
        voltage = voltage[lim:-lim]
        current = current[lim:-lim]
        time = time[lim:-lim]
        # Make sure the arrays have an even number of elements
        if voltage.size % 2 != 0: 
            voltage = np.delete(voltage,0)
            current = np.delete(current,0)
            time = np.delete(time,0)
    return voltage, current, time

#------------------------------------------------------------------------------#

def fourier_transform(voltage, current, SAMPLING_RATE, voltage2=None):
    """
    Compute the Fourier transform of current and voltage signals. 
    A second voltage signal can be trasformed as optional for three-electrodes
    configuration measurement.

    """
    n_samples = voltage.size
    print('Computing FFT...')    
    ft_voltage = fftshift(fft(voltage)) / n_samples
    ft_current = fftshift(fft(current)) / n_samples
    if voltage2 is not None:
        ft_voltage2 = fftshift(fft(voltage2)) / n_samples
    t0 = time.time()
    freq_axis = fftshift(fftfreq(int(n_samples), d=SAMPLING_RATE))
    print('Elapsed time for generating the frequency axis: ' +
          str(time.time()-t0) + ' s')
    print('FFT complited.')   
    if voltage2 is None: 
        return ft_voltage, ft_current, freq_axis
    else:
        return ft_voltage,ft_voltage2, ft_current, freq_axis


#------------------------------------------------------------------------------#


def fermi_dirac_filter(fr, fc, bw, n):
    """
    Create a digital filter shaped as two symmetric Fermi-Dirac functions. 

    """
    X = (fr - fc) / bw
    Y = np.cosh(n) / (np.cosh(n*X) + np.cosh(n))
    return Y


#------------------------------------------------------------------------------#


def find_freq_indexes(input_signal, multisine_freq, SAMPLING_RATE, freq_axis):
    """
    Find the peaks in the frequency domain. Due to instrumantation artifacts the
    frequenies in the multisine can be different then the selected one. 
    The peak is looked for in the range half the distance between the previous 
    and following peak. For frequencies below 1 Hz the searching interval is 
    reduced to a quarter of the distance to the neighboring peaks.
    This algorithm is translated from the orignal script of Alberto Battistel
    https://github.com/alberto-battistel/DMFA
    """
    index_f0 = np.where(freq_axis == 0)[0][0]
    n_samples = input_signal.size
    guess_i_multisine_freq = np.ceil(multisine_freq * SAMPLING_RATE * n_samples) + index_f0
    index_multisine_freq = np.zeros(multisine_freq.size,dtype='int64')
    dist_between_freq = np.zeros(multisine_freq.size, dtype='float32')
    dist_between_freq[0] = np.min([multisine_freq[1] - multisine_freq[0], multisine_freq[0]])
    for f in range(1, multisine_freq.size-1):
        dist_between_freq[f] = np.min([multisine_freq[f] - multisine_freq[f-1], multisine_freq[f+1] - multisine_freq[f]])    
    dist_between_freq[-1] = multisine_freq[-1] - multisine_freq[-2] # Fixed
    # dist_peak = dist_between_freq/(dt*N_samples)
    dist_between_freq_P = 2 * np.floor(dist_between_freq * SAMPLING_RATE  * n_samples / 4)
    for f in range(0, multisine_freq.size):
        if multisine_freq[f] < 0.1:
            min_searching = int(guess_i_multisine_freq[f]) - int(dist_between_freq_P[f]/4)
            max_searching = int(guess_i_multisine_freq[f]) + int(dist_between_freq_P[f]/4)
        else:    
            min_searching = int(guess_i_multisine_freq[f]) - int(dist_between_freq_P[f]/2)
            max_searching = int(guess_i_multisine_freq[f]) + int(dist_between_freq_P[f]/2)
        searching_freq_rng =  np.arange(min_searching, max_searching)
        index_multisine_freq[f] = min_searching + int(np.argmax((np.abs(input_signal[(searching_freq_rng)])),axis = 0))   
    return index_multisine_freq


#------------------------------------------------------------------------------#


def extract_impedance(ft_voltage, ft_current, multisine_freq, Npts_elab, index_multisine_freq, SAMPLING_RATE, DT, bw, n):
    """
    Compute the impedance transfer function (time domain) for each frequency 
    dividing the filtered peak (frequency domain) of the voltage by current.
    This algorithm is based on the work of Alberto Battistel and Fabio La Mantia,
    more can be found their pulications:
    [1] Dynamic impedance spectroscopy using dynamic multi-frequency 
        analysis: A theoretical and experimental investigation - Koster et al. 2017
    [2] On the physical definition of dynamic impedance: How to design 
        an optimal strategy for data extraction - Battistel, La Mantia, 2019
    """
    # Prepare the needed parameters
    dist_between_freq = np.zeros(multisine_freq.size, dtype='float32')
    dist_between_freq[0] = np.min([multisine_freq[1] - multisine_freq[0], multisine_freq[0]])
    for f in range(1, multisine_freq.size-1):
        dist_between_freq[f] = np.min([multisine_freq[f] - multisine_freq[f-1], multisine_freq[f+1] - multisine_freq[f]])    
    dist_between_freq[-1] = multisine_freq[-1] - multisine_freq[-2] # Fixed for the last frequency
    impedance = np.zeros((multisine_freq.size, Npts_elab), dtype='complex64')
    # Compute impedance Z(t) for each frequency
    for f in range(0, multisine_freq.size):
        # make sure that the elaboration range is ???
        if Npts_elab%2==0:
            elaboration_rng = np.arange(index_multisine_freq[f] - int(Npts_elab/2), index_multisine_freq[f] + int(Npts_elab/2), 1)
        else:
            elaboration_rng = np.arange(index_multisine_freq[f] - math.floor(Npts_elab/2),index_multisine_freq[f] + math.ceil(Npts_elab/2), 1)
        fd_filter = fermi_dirac_filter(np.linspace(-1/(2*DT), 1/(2*DT), Npts_elab),0,bw,n)
        voltage_portion = Npts_elab * ifft(ifftshift(ft_voltage[elaboration_rng] * fd_filter))
        current_portion = Npts_elab * ifft(ifftshift(ft_current[elaboration_rng] * fd_filter))
        impedance[f] = voltage_portion / current_portion
    print('Impedance extracted.')
    return impedance


#------------------------------------------------------------------------------#


def extract_zero_freq(ft_voltage, ft_current, freq_axis, Npts_elab, SAMPLING_RATE, DT, bw, n):
    """
    Extract the main perturbation.

    """
    index_f0 = np.where(freq_axis == 0)[0][0] # Zero-frequency index
    # Npts_elab = math.ceil(N_samples/N_impedances)
    fd_filter = fermi_dirac_filter(freq_axis[index_f0] + np.linspace(-1/(2*DT), 1/(2*DT), Npts_elab), 0, bw, n)
    freq_range = np.linspace(index_f0 - int(Npts_elab/2),index_f0 + int(Npts_elab/2), Npts_elab, dtype = 'int64')
    V0 = Npts_elab * ifft(ifftshift(ft_voltage[freq_range]*fd_filter)).real
    I0 = Npts_elab * ifft(ifftshift(ft_current[freq_range]*fd_filter)).real       
    time_experiment = DT * np.arange(Npts_elab)
    print('Zero-frequency extracted.')
    return V0, I0, time_experiment


#------------------------------------------------------------------------------#


def baseline_remover(signal):
    """
    Create a straight line between the first and the last points of a signal.
    The line values are stored in an array of the same lenght of the signal.
    """
    xstart = 0
    ystart = signal[0]
    xfinish = signal.size
    yfinish = signal[-1]
    m = (yfinish - ystart) / (xfinish - xstart)
    q = yfinish - (m * xfinish)
    line = np.arange(xstart, xfinish, 1) * m + q 
    return signal - line


#------------------------------------------------------------------------------#


def redo_baseline(original_signal, signal):
    """
    Re-apply the baseline to a signal.
    """
    xstart = 0
    ystart = original_signal[0]
    xfinish = signal.size
    yfinish = original_signal[-1]
    m = (yfinish - ystart) / (xfinish - xstart)
    q = yfinish - (m * xfinish)
    line = np.linspace(xstart, xfinish, signal.size)*(m) + q
    signal = signal + line   
    return signal
    

#------------------------------------------------------------------------------#

def plot_signals(voltage, current, time, text, downsample):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plt.suptitle(text+f'\nDownsampled signals (1 points every {downsample})')
    ax1.plot(time[::downsample]/60, voltage[::downsample])
    ax1.set(xlabel='Time (min)', ylabel='Voltage (V)')
    ax2.plot(time[::downsample]/60, current[::downsample])
    ax2.set(xlabel='Time (min)', ylabel='Current (A)')
    
    
#------------------------------------------------------------------------------#


def slider_plot_impedance(impedance):
    """
    Create a figure with a slider to explore the time-varying impedance interactively. 
    """
    #Initialize figure
    fig = plt.figure()
    ax = fig.subplots()
    start_index = int(impedance[1].size/2)
    p, = ax.plot(impedance[:,start_index].real,- impedance[:,start_index].imag,'-o') 
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.axis('equal')
    ax.set_xlabel('$Z_{real} / Ohm$',fontsize=12)
    ax.set_ylabel('$-Z_{imag} / Ohm$',fontsize=12)
    ax.grid()
    '''comma is important because: As per the documentation for plot(), it returns
    a list of Line2D object. If you are only plotting one line at a time, you can 
    unpack the list using (notice the comma) '''
    plt.subplots_adjust(bottom=0.25)
    #Define updater functions
    # The function to be called anytime a slider's value changes
    def update(val):
        val = val-1
        i = index.val-1
        p.set_xdata(impedance[:,i].real)
        p.set_ydata(-impedance[:,i].imag)
        fig.canvas.draw()    
    def reset_view(val):
        ax.relim()
        ax.autoscale_view()
    # Index slider config and call
    ax_slide = plt.axes([0.1, 0.05, 0.65, 0.03])
    index = Slider(
        ax=ax_slide,
        label='Index',
        valmin=1,
        valmax=impedance[1].size,
        valinit= start_index,
        valfmt = '%d',
        valstep= 1
        )
    index.on_changed(update)
    # Reset button config and call
    axes = plt.axes([0.82, 0.03,  0.15, 0.085])
    breset = Button(axes, 'Reset view', hovercolor='tab:blue')
    breset.on_clicked(reset_view)
    return index, breset


#------------------------------------------------------------------------------#

# MAIN CODE

if __name__ == '__main__':
    # Define measurement parameters
    sampling_rate = 250e-6 # s
    multisine_freq = np.loadtxt('freqs_1k-10m_8ptd.txt') # Frequencies in the multi-sine perturbation
    # Load sampled signals
    voltage = np.load('voltage_charge.npy')
    current = np.load('current_charge.npy')
    time_experiment = np.load('time.npy')
    plot_signals(voltage, current, time_experiment, 'Original signals', 1000)
    voltage, current, time_experiment = cut_integer_periods(voltage, 
                                                            current, 
                                                            time_experiment, 
                                                            multisine_freq, 
                                                            sampling_rate)
    plot_signals(voltage, current, time_experiment, 'Signals cut for an integer number of multisines', 1000)
    # Removing baseline to reduce the drift skirt
    voltage_original = np.copy(voltage)
    voltage = baseline_remover(voltage)
    plot_signals(voltage, current, time_experiment, 'Voltage baseline removed', 1000)
    # Computing fft of the signals
    ft_voltage, ft_current, freq_axis = fourier_transform(voltage,
                                                          current,
                                                          sampling_rate)
    # Optional: calculate the position of the peaks in Fourier space
    index_multisine_freq = find_freq_indexes(ft_current,
                                             multisine_freq,
                                             sampling_rate,
                                             freq_axis)                                                                       
    # Define parameters for DMFA estraction
    n = 8
    bw = 0.01
    DT = 100
    total_exp_time = voltage.size * sampling_rate
    Npts_elab = math.floor(total_exp_time/DT)
    print(f'Number of impedance spectra: {Npts_elab}')
    # Performe the DMFA to extract the impedance and drift
    impedance = extract_impedance(ft_voltage, 
                                  ft_current,
                                  multisine_freq, 
                                  Npts_elab, 
                                  index_multisine_freq, 
                                  sampling_rate, 
                                  DT, 
                                  bw, 
                                  n)
    index, breset = slider_plot_impedance(impedance)
    V0, I0, time_dmfa = extract_zero_freq(ft_voltage, 
                                          ft_current, 
                                          freq_axis, 
                                          Npts_elab, 
                                          sampling_rate, 
                                          DT, 
                                          bw, 
                                          n)
    # Apply again the removed baseline on the extracted voltage
    V0 = redo_baseline(voltage_original, V0)
    plot_signals(V0, I0, time_dmfa, 'Signals extracted with DMFA', 1)
