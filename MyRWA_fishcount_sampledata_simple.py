"""
MyRWA_fishcount_sampledata_sklearnGP.py

Thus

Author: Nathan Sanders
Date: 1/2/2016
Version: 0.1
"""

################################
### Imports
################################

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.ticker import MultipleLocator


################################
### Settings
################################

max_count = 2000 # Maximum possible count in a 10 minute span
false_flag = 1/100. # false positives per minute
max_video_duration = 1. # Maximum video duration, in minutes
inactivity_timer = 4/60. # Timeout after last fish motion detected to stop video, in minutes


################################
### Load data
################################

## 2016 data
historic_data = pd.read_excel('herring 2016 data FINALa.xlsx') 
## 2015 data
#historic_data = pd.read_excel('Raw_Data_6.xlsx')

historic_data['startdate'] = historic_data.apply(lambda x: pd.Timestamp(str(x['Date']).split()[0] + ' ' + str(x['Start_Time'])), axis=1)
historic_data = historic_data.sort('startdate')
historic_data['timedelta'] = (historic_data['startdate'] - historic_data['startdate'].iloc[0]).astype(int) / (1e9 * 60 * 10) # 10min intervals

historic_data.index = historic_data['timedelta']

################################
### Define rate model
################################

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

## Assign random crossing times to each fish
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


################################
### Generate synthetic videos
################################

video_starts = [t_rep_comb[0],]
video_ends = []
video_counts = [t_rep_comb_key[0],]
types = {i:0 for i in range(8)}

## Step through fish
for i in range(1, len(t_rep_comb)-1):
	
	time_of_last_fish = t_rep_comb[i-1]
	time_of_current_fish = t_rep_comb[i]
	time_of_next_fish = t_rep_comb[i+1]
	count_of_current_fish = t_rep_comb_key[i]
	time_current_video_started = video_starts[-1]
	
	#### Logical triggers
	
	## Will we surpass the max video length after the inactivity_timer?
	if (time_of_current_fish - time_current_video_started + inactivity_timer) > max_video_duration:
		video_overrun = True
	else:
		video_overrun = False
	
	## Is there another fish coming soon?
	if (time_of_next_fish - time_of_current_fish) > inactivity_timer:
		fish_coming_soon = False
	else:
		fish_coming_soon = True
	
	## Is this coming right after some other fish?
	if (time_of_current_fish - time_of_last_fish) < inactivity_timer:
		fish_came_right_before = True
	else:
		fish_came_right_before = False

	turnover_time = time_current_video_started + max_video_duration
	
	#### Update simulation
	#### Guiding rule: only start a video at the smae time you end the previous one
	if video_overrun is True and fish_coming_soon is True:
		if fish_came_right_before:
			types[0] += 1
			## At the end of a video in a stream of fish
		
			## Does this fish go in the current video or the next one?
			if (time_of_current_fish - time_current_video_started) > max_video_duration:
				## Next one
				video_counts += [count_of_current_fish, ]
			else:
				## Current one
				video_counts[-1] += count_of_current_fish
				video_counts += [0, ]
			
			video_ends += [turnover_time, ]
			video_starts += [turnover_time, ]
		
		else:
			types[1] += 1
			## start of a new video
			video_ends += [time_of_last_fish + inactivity_timer, ]
			video_starts += [time_of_current_fish, ]
			video_counts += [count_of_current_fish, ]
	
	elif video_overrun is True and fish_coming_soon is False:
		if fish_came_right_before:
			types[2] += 1
			## A leftover fish
			video_ends += [turnover_time, ]
			video_starts += [turnover_time, ]
			
			## Does this fish go in the current video or the next one?
			if (time_of_current_fish - time_current_video_started) > max_video_duration:
				## Next one
				video_counts += [count_of_current_fish, ]
			else:
				## Current one
				video_counts[-1] += count_of_current_fish
				video_counts += [0, ]
			
		else:
			types[3] += 1
			## An isolated fish
			video_counts += [count_of_current_fish, ]
			video_ends += [time_of_last_fish + inactivity_timer, ]## NOTE: Do we need a 3s leftover video?
			video_starts += [time_of_current_fish, ]
		
	elif video_overrun is False and fish_coming_soon is True:
		if fish_came_right_before:
			types[4] += 1
			## In the middle of a video
			video_counts[-1] += count_of_current_fish
		else:
			types[5] += 1
			## Start of a new stream
			video_ends += [time_of_last_fish + inactivity_timer, ]
			video_starts += [time_of_current_fish, ]
			video_counts += [count_of_current_fish, ]
		
	elif video_overrun is False and fish_coming_soon is False:
		if fish_came_right_before:
			types[6] += 1
			## At the end of a short video
			#video_ends += [time_of_current_fish + inactivity_timer, ] ## Will end at next video
			video_counts[-1] += count_of_current_fish
		else:
			types[7] += 1
			## Short isolated video
			video_ends += [time_of_last_fish + inactivity_timer, ]
			video_starts += [time_of_current_fish, ]
			video_counts += [count_of_current_fish, ]

## Finish off video_ends
## Can assume this will be an isolated fish
video_ends += [video_starts[-1] + inactivity_timer, ]



################################
### Make plot
################################

video_starts = np.array(video_starts)
video_ends = np.array(video_ends)
video_counts = np.array(video_counts)

#def interleave(*args):
	#c = np.empty(sum([len(a) for a in args]), dtype=a.dtype)
	#for i,arg in enumerate(args):
		#c[i::len(args)] = arg
	#return c

plt.figure(figsize=(10,6))
plt.plot(X_mod_min * 10, c_rep_min, lw=1, label='Simulated fish counts per minute', 
	 marker='o', mec='none', color='.5', drawstyle='steps-mid')
#plt.plot(interleave(video_starts, video_ends, video_ends), 
	 #interleave(np.zeros(len(video_starts)), video_counts, np.zeros(len(video_starts))),
	 #lw = 0.5, label='Counts per video', ls='dashed', color='g')
plt.plot(video_starts, video_counts,
	 drawstyle='steps-pre', lw = 0.5, label='Counts per video', ls='dashed', color='g')
plt.legend(prop={'size':8})
plt.xlabel('Time in season (minutes)')
plt.ylabel('Fish counts')
plt.savefig('fish_counts_minute_sim.png', dpi=300)

## Add points and vertical lines
plt.plot(video_starts, video_counts,
	 marker = 'o', mec = 'none', ms=4,
	 drawstyle='steps-pre', lw = 0.5, label='Counts per video', ls='dashed', color='g')

plt.axis([24508, 24590, 0, 4.3])
plt.savefig('fish_counts_minute_sim_zoom1.png', dpi=300)
plt.axis([60000, 60041, 0, 15])
plt.savefig('fish_counts_minute_sim_zoom2.png', dpi=300)
plt.axis([70000, 70041, 0, 40])
plt.savefig('fish_counts_minute_sim_zoom3.png', dpi=300)

## Time duration histogram
plt.figure()
plt.hist((video_ends - video_starts) * 60, 100, color='.5', edgecolor='none', log=1)
plt.ylabel('Number of videos')
plt.xlabel('Duration (s)')
plt.gca().xaxis.set_minor_locator(MultipleLocator(1)) 
plt.savefig('fish_counts_histogram.png', dpi=300)


################################
### Write out
################################

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
video_df[['Start Time (min)', 'End Time (min)', 'Duration (s)', 'True Fish Count', 'Start Timestamp', 'End Timestamp']].to_csv('simulated_fish_videos.csv', index=0)

## Print out individual fish crossing times
pd.Series(np.round(t_rep_min, 2)).to_csv('simulated_fish_crossings.csv', index=0)



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
		self.assertTrue(res <= max_video_duration)
        
	def test_blanks_exist(self):
		self.assertTrue(np.any(video_df['True Fish Count'] == 0) or false_flag == 0)

	def test_correct_count(self):
		self.assertEqual(video_df['True Fish Count'].sum().astype(int), t_rep_comb_key.sum().astype(int))


if __name__ == '__main__':
    unittest.main()




