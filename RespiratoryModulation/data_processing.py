import matplotlib as mpl; mpl.use('Agg')
from matplotlib import pyplot as plt
from scipy import signal
from scipy import optimize
import numpy as np
import pandas as pd
import toml
import pickle

def make_smooth_derivative( vent, w1 = 10, w2 = 10 ):
    '''compute and smooth derivative of ventilation'''
    size_for_diff = vent.size - ( 2 * w1 )
    diff = np.zeros(size_for_diff)
    for i in np.arange(size_for_diff):
        diff[i] = vent[i+( 2 * w1 )] - vent[i]
    size_for_median_diff = diff.size - ( w2 * 2 )
    median_diff = np.zeros(size_for_median_diff)
    for i in np.arange( size_for_median_diff ):
        median_diff[i] = np.median( diff[i:i+(2 * w2)] )
    return median_diff

def load_parse_data( subject_idx ):
    '''read text file for subject_idx into dataframe and make some useful column names'''
    df = pd.read_csv('data/s'+str(subject_idx)+'.txt', delimiter='\t')
    channel_names = list(df.keys())
    new_names = list(channel_names)
    my_names=['ecg','e-trig','i-trig']    
    for i,channel_name in enumerate(channel_names):
        for my_name in my_names:                
            ln = channel_name.lower()
            lnf = ln.find(my_name)
            if lnf > -1:
                new_names[i] =  my_name
                break
            if ln == '5 ':
                new_names[i] = 'ap'
                break
            elif ln =='3 ':
                new_names[i] = 'resp'
                break
    new_name_dict = dict(zip(df.keys(),new_names))
    df.rename( columns=new_name_dict, inplace = True)
    return df
        
def get_putative_events( md, df, subject_idx, td_thresh, i_thresh = 0, e_thresh = 0, w1 = 10, w2 =10 ):

    putative_event_i_thresh=[]
    putative_event_e_thresh=[]
    last_i = -2
    last_e = -1
    for i in range(md.size-1):
        i_thresh_lower_bound = md[i+1] > i_thresh
        i_thresh_upper_bound = md[i] <= i_thresh
        if i_thresh_lower_bound and i_thresh_upper_bound:
            if last_i > last_e:
                continue
            else:
                putative_event_i_thresh.append(i)
                last_i = i
                continue
        e_thresh_upper_bound = md[i+1] < e_thresh
        e_thresh_lower_bound = md[i] >= e_thresh
        if e_thresh_upper_bound and e_thresh_lower_bound:
            if last_e > last_i:
                continue
            else:
                putative_event_e_thresh.append(i)
                last_e = i

    vent = df['resp']
    zip_ie = zip( putative_event_i_thresh,
                  putative_event_e_thresh )
    putative_event_td_diff = [ vent[e+w2] - vent[i+w2] for i,e in zip_ie ]
    zip_i_e_td = zip( putative_event_i_thresh,
                      putative_event_e_thresh,
                      putative_event_td_diff )
    confirmed_cycle_list = [ (i,e) for i,e,td in zip_i_e_td
                             if td > td_thresh ]
    ie_cycles = np.array( confirmed_cycle_list )
    t_shift_w2 = df['Time'][w2]
    ie_trigger_times = df['Time'].values[ie_cycles] + t_shift_w2
    ie_trigger_path = 'data/ie_s'+str( subject_idx )
    np.save( ie_trigger_path, ie_trigger_times )
    return ie_trigger_times

def process_ap( ap ):
    diff_ap = np.diff(ap)
    h5,_ = signal.find_peaks( diff_ap,
                              height = 5,
                              distance = 15, )

    systole=np.zeros(h5.size-1).astype(int)
    diastole = np.zeros(h5.size-1).astype(int)
    for h in np.arange(h5.size-1):
        systole[h] = h5[h] + np.argmax(ap[ h5[h]:h5[h+1] ])
        diastole[h]= h5[h] + np.argmin(ap[ h5[h]:h5[h+1] ])
        
    return systole,diastole

def process_model_ap(ap):
    ht = np.mean(ap)
    s,_ = signal.find_peaks( ap, height = ht, distance = 20, )
    d=[]
    for i in np.arange(s.size-1):
        d.append(
            s[i]+np.argmin( ap[ s[i] : s[i+1] ] )
        )
    s=s[1:]
    return np.array(s),np.array(d)

def get_epoch_avgbp ( s_times, s_indecise, d_times, d_indecise, ap):
    # make sure we have the pairs that we _expect_
    # there should be a diastole before each systole
    # there should be the same number of systoles and diastoles
    if d_times[0] > s_times[0]:
        s_t=s_times[1:]
        s_i=s_indecise[1:]
    else:
        s_t=s_times
        s_i=s_indecise
    if d_times[-1] < s_times[-1]:
        d_t=d_times
        d_i=d_indecise
    else:
        d_t=d_times[:-1]
        d_i=d_indecise[:-1]
    # now average each diastole to the following
    # systole with the weight of 2 to 1
    # this is Mean Arterial Pressure
    # but it is called "avgbp" in these codes
    avgbp_times = d_t 
    avgbp = ( ap[s_i]+( 2* ap[d_i] ) ) / 3.
    pulse_pressure = ap[s_i] - ap[d_i]
    return avgbp_times, avgbp, pulse_pressure

def get_interval( u, tt ):
    i1 = u > tt[0]
    i2 = u <= tt[1]
    i3 = i1 * i2
    return( u[i3] )

def select_epoch_data( df, ie_trigger_times, epochs, subject_idx ):
    # this method will select from data the intervals for our three experimental epochs    
    ecg_times = df['Time'][(df['ecg']>0).values].values # the time of each R peak
    # epoch_blah has values of blah
    # epoch_blah_time has the times at which blah occurs
    # epoch_blah_idx has index values for each blah in rows of df or in array
    epochs_metrics = dict()
    epochs_ecg_times = dict()
    epochs_e_times = dict()
    for epoch_name, epoch_times in epochs.items():
        # epoch_times has two elements; the start time and the end time of the epoch        
        i1 = (df['Time'] > epoch_times[0]).values 
        i2 = (df['Time'] <= epoch_times[1]).values
        i3 = i1 * i2
        epoch_time = df['Time'][i3].values
        epoch_ap = df['ap'][i3].values
        epoch_systole_idx, epoch_diastole_idx = process_ap(epoch_ap )
        epoch_systole_times = epoch_time[epoch_systole_idx]
        epoch_diastole_times = epoch_time[epoch_diastole_idx]        
        # TODO get epoch_systole_times
        epoch_avgbp_times, epoch_avgbp,epoch_pp = get_epoch_avgbp(
            epoch_systole_times, epoch_systole_idx,
            epoch_diastole_times, epoch_diastole_idx,
            epoch_ap
        )
        # pay attention "epoch_ecg_times_" gets chopped up
        # and turned into "epoch_ecg_times"
        epoch_ecg_times_ = get_interval( ecg_times, epoch_times )
        epoch_rri = np.diff( epoch_ecg_times_ )
        epoch_rri_times = (epoch_ecg_times_[:-1] + epoch_ecg_times_[1:])/2
        epoch_ecg_times = epoch_ecg_times_[:-1] # correct size
        # second collum of ie_trigger_times is the beginning of expiration
        epoch_e_times = get_interval( ie_trigger_times[:,1], epoch_times )
        epochs_metrics[epoch_name] = dict()
        epochs_metrics[epoch_name]['rri'] =     [epoch_rri_times,   epoch_rri ]
        epochs_metrics[epoch_name]['avgbp'] =   [epoch_avgbp_times, epoch_avgbp ]
        epochs_metrics[epoch_name]['pp'] =      [epoch_avgbp_times, epoch_pp]
        epochs_e_times[epoch_name] = epoch_e_times
        epochs_ecg_times[epoch_name] = epoch_ecg_times
    return epochs_metrics , epochs_e_times, epochs_ecg_times

def get_cycle_metrics( salient_times,
                       related_array,
                       e1, e2 ):
    cycle_lower_bound = salient_times > e1
    cycle_upper_bound = salient_times <=e2
    cycle_indexes = cycle_lower_bound * cycle_upper_bound
    cycle_times = salient_times[cycle_indexes]
    cycle_related = related_array[cycle_indexes]
    if cycle_related.size:
        cycle_phase = (cycle_times - e1)/(e2-e1)
        return np.hstack(
            ( cycle_phase.reshape((cycle_phase.size,1)),
              cycle_related.reshape((cycle_phase.size,1)) )
        )

def compute_respiratory_modulation_coefficients(epochs_metrics, epochs_e_times, epochs, subject_idx, results_key=['rri','avgbp','pp'], angle_range=[[0,2*np.pi],[-np.pi,np.pi],[0,2*np.pi]]):
    # this method does _not_ know if it is operating on model output or human data
    epoch_results_accumulator = dict()    
    epoch_cycle_results = dict()
    modulation_coefficients = dict()
    def cosine_tune( x, b, c, t ):
        return b + c *np.cos(x*np.pi*2-t)
    for epoch_name, epoch_times in epochs.items():
        epoch_e_times = epochs_e_times[epoch_name]
        epoch_metrics = epochs_metrics[epoch_name]
        for key in results_key:
            epoch_results_accumulator[key] = list()
            
        for j in range(epoch_e_times.size-1):
            e1 = epoch_e_times[j]
            e2 = epoch_e_times[j+1]
            for key in results_key:
                this_cycle = get_cycle_metrics(
                    *epoch_metrics[key],
                    e1, e2 )
                # keep in mind that there may be more than one
                # cardiovascular event in each respiratory cycle
                # so we will np.concatenate the accumulated results in a moment
                if type(this_cycle) != type(None):
                    epoch_results_accumulator[key].append(this_cycle)
        for key in results_key:
            # as promised
            epoch_cycle_results[key] = np.concatenate(epoch_results_accumulator[key] )

        modulation_coefficients[epoch_name] = dict()            
        angle_dict = dict( zip( results_key, angle_range) )
        for key in results_key:
            # i'm not putting any constraints on the modulation amplitude or phase
            # amplitude could end up positive or negative, and separately,
            # phase could be "off" by integer multiples of 2 pi.
            # so we are later check and correct the amplitude,phase for each case 
            # in order to make meaningful comparisons between epochs.
            try:
                popt,pcov = optimize.curve_fit(
                    cosine_tune,
                    epoch_cycle_results[key][:,0],
                    epoch_cycle_results[key][:,1],
                    [0,1,np.pi/2] )
            except RuntimeError:
                # this should only trigger when analyzing pp data from model 1
                # which doesn't actually change, so the optimization throws
                # an runtime failure when it fails to fit. 
                print('curve fit runtime error:',epoch_name,key)
                print('substituting mean of metric')
                popt = np.array([ np.mean(epoch_cycle_results[key][:,1]), 0, 0])
                # check coefficients and correct if necessary (see previous comment)
            define_angle = angle_dict[key]
            corrected_angle = get_standardized_polar_coord( popt[2], popt[1],
                                                            define_angle )
            
            corrected_coefficients = np.array([ popt[0], np.abs(popt[1]),
                                                corrected_angle ])
            pickle.dump( (epoch_cycle_results[key][:,0], epoch_cycle_results[key][:,1]),
                         open(f'results_output/tuple_{key}_processed_data_{epoch_name}_{subject_idx}.pickle','wb'))
            modulation_coefficients[epoch_name][key] = corrected_coefficients
    return modulation_coefficients

def rectify_d( b, angle_range ):
    if b > angle_range[1]:
        while(b > angle_range[1]):
            b = b - 2*np.pi
    if b < angle_range[0]:
        while(b < angle_range[0]):
            b = b + 2*np.pi
    return b

def get_standardized_polar_coord(tp,bp,define_angle):
    # tp is phase offset angle
    # bp is modulation depth
    # define_angle is the 2*pi sized range of angles 
    d = tp # d for modulation _D_irection, which is an angle.
    m = bp # m for _M_odulation depth. 
    if m < 0:
        m = -m
        d = d-np.pi
    d = rectify_d( d, define_angle )
    return np.array(d)

def synthesize_modulated_pulse_pressure( epoch_ecg_times, epoch_e_times, epoch_pp_mod_coeff ):
    # across all volunteers and all epochs, the blood pressure event is
    # between 270 and 330 ms after the R-peak
    # so we are using R-peak time plus 300 ms for blood pressure events
    # in our models. 

    shifted_ecg_times = epoch_ecg_times + 0.3
    pp_popts = epoch_pp_mod_coeff
    pp_amplitude_list = []
    pp_time_list = []
    for i in range(epoch_e_times.size - 1):
        e1 = epoch_e_times[i]
        e2 = epoch_e_times[i+1]
        ediff = e2 - e1
        this_idx = (shifted_ecg_times > e1) * (shifted_ecg_times <=e2)
        ecg_this_cycle = shifted_ecg_times[ this_idx ]
        for hb in ecg_this_cycle:
            phase = (hb-e1) / ediff
            pp = (pp_popts[1] * np.cos( phase*np.pi*2 - pp_popts[2] )) + pp_popts[0]
            pp_amplitude_list.append( pp )
            pp_time_list.append( hb )

    return np.array(pp_amplitude_list), np.array(pp_time_list)

def models( epochs_ecg_times, epochs_e_times, modulation_coefficients):

    def simulate_bp( pp ):
        pp_a = pp[0]
        pp_t = ( (pp[1] ) * 1000).astype(np.int)
        ap_len = pp_t[-1] - pp_t[0]
        stroke = 0
        x=p1
        p = np.zeros(ap_len)
        n=0
        last_t=0
        for t,a in zip( pp_t[1:]-pp_t[0], pp_a[:-1]):
            stroke=a
            for j in np.arange(t-last_t):
                if stroke > 0:
                    x += stroke
                else:
                    x +=  ( - (x-p0)) / taup
                p[n] = x
                n+=1
                stroke=0
            last_t=t
        return p,pp[1][0]

    epoch_names = modulation_coefficients.keys()
    model1_metrics = dict()
    model2_metrics = dict()
    for epoch_name in epoch_names:
        epoch_pp_mod_coeff = modulation_coefficients[epoch_name]['pp']
        epoch_pp_amplitude, epoch_pp_times = synthesize_modulated_pulse_pressure(
            epochs_ecg_times[epoch_name], epochs_e_times[epoch_name], epoch_pp_mod_coeff )
        
        # model#2
        p2 =110
        p1 = 50
        hr=.0008
        taup=1200
        p0=(p1-p2*np.exp(-1./hr/taup))/(1-np.exp(-1./hr/taup));
        dt = 1
        epoch_model2_ap, model2_t0 = simulate_bp( (epoch_pp_amplitude, epoch_pp_times) )
        model2_s,model2_d = process_model_ap( epoch_model2_ap )

        model2_time =  np.arange(epoch_model2_ap.size)/1000 + model2_t0
        s_times = model2_time[model2_s]
        d_times = model2_time[model2_d]        
        model2_avgbp_times, model2_avgbp, model2_pp =get_epoch_avgbp(
            s_times, model2_s,
            d_times, model2_d,
            epoch_model2_ap )
        model2_metrics[epoch_name] = dict()
        model2_metrics[epoch_name]['avgbp']  =   [model2_avgbp_times, model2_avgbp ]
        model2_metrics[epoch_name]['pp']  =   [model2_avgbp_times, model2_pp ]
        
        # model#1
        p2=135
        p1=60
        hr=.0008
        taup=750
        p0=(p1-p2*np.exp(-1./hr/taup))/(1-np.exp(-1./hr/taup));
        dp = p2 - p1
        dt = 1
        pp_fixed_amplitdue = np.empty(epoch_pp_amplitude.size)
        pp_fixed_amplitdue.fill(dp)
        epoch_model1_ap, model1_t0 = simulate_bp( (pp_fixed_amplitdue, epoch_pp_times) )
        model1_s,model1_d = process_model_ap( epoch_model2_ap )        
        model1_time =  np.arange(epoch_model1_ap.size)/1000 + model1_t0
        s_times = model1_time[model1_s]
        d_times = model1_time[model1_d]        
        model1_avgbp_times, model1_avgbp, model1_pp =get_epoch_avgbp(
            s_times, model1_s,
            d_times, model1_d,
            epoch_model1_ap )
        model1_metrics[epoch_name] = dict()
        model1_metrics[epoch_name]['avgbp']  =   [model1_avgbp_times, model1_avgbp ]
        model1_metrics[epoch_name]['pp']  =   [model1_avgbp_times, model1_pp ]
    return model1_metrics , model2_metrics

def data_processing( subject_idx, epochs, td_thresh = 200, compute_ie_times = True ):
    # start here
    # subject_idx specifies the integer in the data file name
    # epochs is a dictionary that specifies the time intervals for analysis in each of the three epochs
    # example:
    # {'baseline': [800, 1700],
    #  'post': [3300, 4100],
    #  'sdb': [2050, 2900]}

    df = load_parse_data(subject_idx)
    if compute_ie_times == True:
        median_diff = make_smooth_derivative( df['resp'].values )
        ie_trigger_times = get_putative_events( median_diff, df, subject_idx, td_thresh )
    else:
        ie_trigger_times = np.load(f'data/ie_s{subject_idx}.npy')

    human_metrics, epochs_e_times, epochs_ecg_times= select_epoch_data(
        df, ie_trigger_times, epochs, subject_idx )

    print('human compute modulation coefficients')                                
    human_modulation_coefficients = compute_respiratory_modulation_coefficients(
        human_metrics, epochs_e_times,epochs, subject_idx )


    model1_metrics, model2_metrics = models( epochs_ecg_times, epochs_e_times,
                                             human_modulation_coefficients )

    print('model1 compute modulation coefficients')
    model1_modulation_coefficients = compute_respiratory_modulation_coefficients(
        model1_metrics, epochs_e_times,epochs, subject_idx,
        results_key=['avgbp','pp'], angle_range = [[-np.pi, np.pi],[0,2*np.pi]] )
    print('model2 compute modulation coefficients')                                
    model2_modulation_coefficients = compute_respiratory_modulation_coefficients(
        model2_metrics, epochs_e_times,epochs, subject_idx,
        results_key=['avgbp','pp'], angle_range = [ [-np.pi, np.pi],[0,2*np.pi]] )
                                                                                  
    return human_modulation_coefficients, model1_modulation_coefficients, model2_modulation_coefficients
