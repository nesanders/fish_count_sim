"""
MyRWA_fishcount_sampledata_simple_discrete.py

Thus

Author: Nathan Sanders
Date: 1/28/2016
Version: 0.4
"""

################################
### Imports
################################

print "Imports"

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.ticker import MultipleLocator
import os


################################
### Settings
### Script always uses unites of minutes except when explicitly specified
################################

max_count = 2000 # Maximum possible count in a 10 minute span
false_flag = 1/100. # false positives per minute
max_video_duration = 1. # Maximum video duration, in minutes
inactivity_timer = 20/60. # Timeout after last fish motion detected to stop video, in minutes
min_fish_crossing_time = 1 / 60. # Minimum time duration that a fish will spend in the frame, in minutes
discretiziation_time = 0.1 / 60. # Unit for discretizing the simulated time grid, in minutes
out_dir = 'run_'+str(int(inactivity_timer * 60))+'s_inactivity/'

def crossing_time_generator(N): 
	# Distribution of crossing times, representing how long each fish stays in the frame
	out = np.random.lognormal(np.log(6),.5, N) / 60. # convert to minutes
	out[out < min_fish_crossing_time] = min_fish_crossing_time
	return out


################################
### Load data
################################

print "Load data"

## 2016 data
historic_data = pd.read_excel('herring 2016 data FINALa.xlsx') 
## 2015 data
#historic_data = pd.read_excel('Raw_Data_6.xlsx')

historic_data['startdate'] = historic_data.apply(lambda x: pd.Timestamp(str(x['Date']).split()[0] + ' ' + str(x['Start_Time'])), axis=1)
historic_data = historic_data.sort('startdate')
historic_data['timedelta'] = (historic_data['startdate'] - historic_data['startdate'].iloc[0]).astype(int) / (1e9 * 60 * 10) # 10min intervals

historic_data.index = historic_data['timedelta']

os.system('mkdir '+out_dir)

################################
### Define rate model
################################

print "Generate fish"

ftrans = lambda x: x
ftransi = lambda x: x

rate_model = interp1d(historic_data['timedelta'].values, historic_data['Count'].values, kind='linear')

## Generate counts every 10 minutes
X_mod_10 = np.arange(historic_data['timedelta'].min(), historic_data['timedelta'].max(), 1)
y_rep_10 = rate_model(X_mod_10)

## Interpolate to every minute
X_mod_min = np.arange(historic_data['timedelta'].min(), historic_data['timedelta'].max(), 0.1)
y_rep_min = pd.Series(ftransi(y_rep_10), index = X_mod_10).ix[X_mod_min].interpolate(method='linear')

## Cap counts at the max_count
y_rep_min[y_rep_min > max_count/10] = max_count/10

## Generate fish counts every minute
c_rep_min = np.random.poisson(np.abs(y_rep_min.values / 10.))

## Assign random entrance times to each fish
t_rep_min = np.empty(np.sum(c_rep_min))
j = 0
for i in range(len(c_rep_min)):
	if c_rep_min[i] > 0:
		t_rep_min[j:j+c_rep_min[i]] = X_mod_min[i]*10 + np.random.uniform(0, 1, c_rep_min[i])
		j += c_rep_min[i]

## Add in false positives at random times
N_false = round(false_flag * len(c_rep_min))
t_rep_min_fp = np.random.uniform(0, len(c_rep_min), N_false)

## Combine real and false positives
t_rep_comb = np.append(t_rep_min, t_rep_min_fp)
t_rep_comb_key = np.append(np.ones(len(t_rep_min)), np.zeros(len(t_rep_min_fp)))

## Mix in the false positives by sorting
ao = np.argsort(t_rep_comb)
t_rep_comb = t_rep_comb[ao]
t_rep_comb_key = t_rep_comb_key[ao]

## Generate random crossing times
t_rep_cross = crossing_time_generator(len(t_rep_comb))

################################
### Generate synthetic videos
################################

min_time = min(t_rep_comb)

print "Generate discretized timesteps"
dis_t = np.arange(min_time, max(t_rep_comb), discretiziation_time)
dis_fish_in_frame = np.zeros(len(dis_t)) ## Is there fish activity at this time?
dis_camera_on = np.zeros(len(dis_t)) ## Is the camera on at this time?

print "Simulate when fish is in frame"
for i in range(len(t_rep_comb)):
	fish_in = t_rep_comb[i]
	fish_out = fish_in + t_rep_cross[i]

	## What time steps is this fish in the frame?
	d_i_in = int(np.floor((fish_in - min_time) / discretiziation_time))
	d_i_out = int(np.floor((fish_out - min_time) / discretiziation_time))
	
	dis_fish_in_frame[d_i_in:d_i_out] = 1

print "Simulate when camera is on"
fish_last_in_frame = -np.inf
for i in range(len(dis_camera_on)):
	## Camera is on if fish is in frame
	if dis_fish_in_frame[i]: 
		dis_camera_on[i] = 1
		fish_last_in_frame = dis_t[i]
	
	## Camera is on if fish was in frame within inactivity_timer
	if (fish_last_in_frame + inactivity_timer) > dis_t[i]:
		dis_camera_on[i] = 1

print "Simulate video start and stop times"
video_starts = [t_rep_comb[0],]
video_ends = []
for i in range(1,len(dis_camera_on)):
	## New video due to turn-on?
	if dis_camera_on[i-1] == 0 and dis_camera_on[i] == 1:
		video_starts += [dis_t[i]]
	
	## End video?
	elif dis_camera_on[i-1] == 1 and dis_camera_on[i] == 0:
		video_ends += [dis_t[i]]
	
	## New video due to overrun?
	elif len(video_starts) > 0 and dis_camera_on[i-1] == 1 and dis_camera_on[i] == 1 and dis_t[i] > (video_starts[-1] + max_video_duration):
		video_ends += [dis_t[i]]
		video_starts += [dis_t[i]]

## finish last video
video_ends += [t_rep_comb[-1] + t_rep_cross[-1]]

print "Measure fish crossing counts"
## Video gets a count if fish entered the frame during its duration
video_counts = []
t_rep_comb_key_b = t_rep_comb_key.astype(bool)
for i in range(len(video_ends)):
	## Note - doing a where and then a len is ~1000x faster than doing a sum over the bools
	sel = np.where((t_rep_comb > video_starts[i]) & (t_rep_comb <= video_ends[i]) & t_rep_comb_key_b)
	video_counts += [len(sel[0])]

video_starts = np.array(video_starts)
video_ends = np.array(video_ends)
video_counts = np.array(video_counts)


################################
### Write out
################################

print "Write outputs"

start_epoch = historic_data['startdate'].min()

video_df = pd.DataFrame({
	'Start Time (min)':np.round(video_starts, 2),
	'End Time (min)':np.round(video_ends, 2),
	'True Fish Count':np.round(video_counts, 2),
	'Duration (s)':np.round((video_ends - video_starts) * 60, 1),
	'Start Timestamp':[start_epoch + np.timedelta64(int(np.floor(d)),'m') + np.timedelta64(int((d-np.floor(d))*60),'s') for d in video_starts],
	'End Timestamp':[start_epoch + np.timedelta64(int(np.floor(d)),'m') + np.timedelta64(int((d-np.floor(d))*60),'s') for d in video_ends],
	})

## Print out video metadata
video_df[['Start Time (min)', 'End Time (min)', 'Duration (s)', 'True Fish Count', 'Start Timestamp', 'End Timestamp']].to_csv(out_dir + 'simulated_fish_videos.csv', index=0)

## Print out individual fish entry and exit times
start_times_real_fish = t_rep_comb[t_rep_comb_key > 0]
pd.DataFrame(data = 
	     {'entrance time': np.round(start_times_real_fish, 2), 'exit time': np.round(start_times_real_fish + t_rep_cross[t_rep_comb_key > 0], 2)}
		     ).to_csv(out_dir + 'simulated_fish_crossings.csv', index=0)


################################
### Make diagnostic plots
################################

print "Make diagnostic plots"

## Plot of simulations
plt.figure(figsize=(10,6))

#plt.plot(interleave(video_starts, video_ends, video_ends), 
	 #interleave(np.zeros(len(video_starts)), video_counts, np.zeros(len(video_starts))),
	 #lw = 0.5, label='Counts per video', ls='dashed', color='g')
plt.plot(video_starts, video_counts,
	 drawstyle='steps-pre', lw = 0.5, label='Counts per video', ls='dashed', color='g')
plt.legend(prop={'size':8})
plt.xlabel('Time in season (minutes)')
plt.ylabel('Fish counts')
plt.savefig(out_dir + 'fish_counts_minute_sim.png', dpi=300)

## Zoomins
## Add points and vertical lines
plt.plot(video_starts, video_counts,
	 marker = 'o', mec = 'none', ms=4,
	 drawstyle='steps-pre', lw = 0.5, ls='dashed', color='g')

for i,i_in,i_out,top in [(1, 24508, 24590, 4.3), (2, 60000, 60041, 15), (3, 70000, 70041, 40)]:
	## Fish appearance
	sel_fish = np.where((t_rep_comb >= i_in) & (t_rep_comb <= i_out) & t_rep_comb_key.astype(bool))[0]
	sel_fp = np.where((t_rep_comb >= i_in) & (t_rep_comb <= i_out) & (t_rep_comb_key.astype(bool)==0))[0]
	for j in sel_fp:
		plt.axvline(t_rep_comb[j], zorder=-5, alpha=0.2, color='k',
			label='False positive' if j==sel_fp[0] else None, lw=0.25)
	for j in sel_fish:
		plt.axvline(t_rep_comb[j], zorder=-5, alpha=0.2, color='r',
			label='Fish appearance' if j==sel_fish[0] else None, lw=0.25)
	
	sel = (dis_t >= i_in) & (dis_t <= i_out)
	## Fish in frame
	plt.plot(dis_t[sel], dis_fish_in_frame[sel], label = 'Fish in frame', color='orange', lw=1)
	## Camera on
	plt.plot(dis_t[sel], dis_camera_on[sel]*0.75, label = 'Camera on', color='b', lw=0.5)
	
	plt.plot(X_mod_min * 10, c_rep_min, lw=1, label='Simulated fish counts per minute', 
		marker='o', mec='none', color='.5', drawstyle='steps-mid')
	
	if i==1: plt.legend(prop={'size':8})
	plt.axis([i_in, i_out, 0, top])
	plt.gca().xaxis.set_minor_locator(MultipleLocator(1))
	plt.savefig(out_dir + 'fish_counts_minute_sim_zoom'+str(i)+'.png', dpi=600)


## Videos per day
plt.figure()
pd.Series(index = video_df['Start Timestamp'], data = np.ones(len(video_df))).groupby(pd.TimeGrouper('1D')).sum().plot(color='k', lw=2, drawstyle='steps-pre')
plt.ylabel('Videos per day')
plt.savefig(out_dir + 'video_counts_perday.png', dpi=300)



## False positive videos per day
plt.figure()
pd.Series(index = video_df['Start Timestamp'], data = (video_df['True Fish Count'] > 0).values.astype(int)).groupby(pd.TimeGrouper('1D')).mean().plot(color='k', lw=2, drawstyle='steps-pre')
plt.ylabel('Fraction of videos that are not false positives (empty with no fish)')
plt.savefig(out_dir + 'video_counts_perday_falsepositive.png', dpi=300)


## Fish counts histogram
plt.figure()
plt.hist(video_counts, np.arange(0, np.max(video_counts)+3) - 0.05, color='.5', edgecolor='none', log=1)
plt.ylabel('Number of videos')
plt.xlabel('# of fish')
plt.gca().xaxis.set_minor_locator(MultipleLocator(1))
plt.gca().set_xlim(0, np.max(video_counts)+3)
plt.savefig(out_dir + 'fish_counts_histogram.png', dpi=300)


## Crossing times histogram
plt.figure()
plt.hist(t_rep_cross * 60., range=[0,40], bins=80, color='.5', edgecolor='none')
plt.ylabel('Number of fish')
plt.xlabel('Crossing duration (s)')
plt.gca().xaxis.set_minor_locator(MultipleLocator(1)) 
plt.savefig(out_dir + 'fish_crossing_times_histogram.png', dpi=300)

## Video duration histogram
plt.figure()
plt.hist((video_ends - video_starts) * 60, np.arange(0, np.ceil(max_video_duration*60 + 10)), color='.5', edgecolor='none', log=1)
plt.ylabel('Number of videos')
plt.xlabel('Video duration (s)')
plt.gca().xaxis.set_minor_locator(MultipleLocator(1)) 
plt.savefig(out_dir + 'video_duration_histogram.png', dpi=300)



################################
### Unit tests
################################

import unittest

class TestVideos(unittest.TestCase):
	#def test_min(self):
		#res = video_df['Duration (s)'].min() / 60.
		#self.assertTrue(res >= inactivity_timer and res > 0)

	def test_max(self):
		res = video_df['Duration (s)'].max() / 60.
		self.assertTrue(res - discretiziation_time <= max_video_duration)
        
        def test_positive_durations(self):
		res = np.any(video_ends < video_starts)
		self.assertTrue(res == False)
        
	def test_blanks_exist(self):
		self.assertTrue(np.any(video_df['True Fish Count'] == 0) or false_flag == 0)

	def test_correct_count(self):
		self.assertEqual(video_df['True Fish Count'].sum().astype(int), t_rep_comb_key.sum().astype(int))
	
	def test_inactivity_gap(self):
		self.assertTrue(np.sum(dis_camera_on) > np.sum(dis_fish_in_frame))


if __name__ == '__main__':
    unittest.main()




