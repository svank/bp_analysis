#!/usr/bin/env python3

import numpy as np
import scipy as scp
import scipy.signal
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.ticker
import pickle
import sys
import os


class PowerSpectralyzer:
    
    plot_period = False
    
    def __init__(self):
        self.time_series = []
        self.weights = []
        self.dt = 1
        
        self.xmin = None
        self.ymin = None
        self.xmax = None
        self.ymax = None
        
        self.new_autocorr = True
    
    def set_dt(self, dt):
        """Sets the delta-t used in computing the frequency values of the fft."""
        self.dt = dt
    
    def add(self, *time_series, weight=None, apodize=False):
        """Add a velocity sequence (in x and y) to the accumulated list."""
        sizes = [a.size for a in time_series]
        if len(np.unique(sizes)) > 1:
            raise RuntimeError("Array sizes should match.")
        if weight is None:
            weight = 1
        
        if apodize:
            window = scipy.signal.windows.hann(sizes[0], False)
            for ts in time_series:
                ts *= window
        
        if len(self.time_series) == 0:
            for ts in time_series:
                self.time_series.append([ts])
        elif len(self.time_series) != len(self.time_series):
            raise RuntimeError(
                    "Number of tiem series should match previous invokations.")
        else:
            for i in range(len(time_series)):
                self.time_series[i].append(time_series[i])
        
        self.weights.append(weight)
    
    @classmethod
    def set_plot_period(cls, on=True):
        cls.plot_period = on
    
    def downsample(self, factor):
        for i in range(len(self.time_series)):
            for j in range(len(self.time_series[i])):
                self.time_series[i][j] = self.time_series[i][j][::factor]
    
    def set_bounds(self, xmin=None, xmax=None, ymin=None, ymax=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
    
    def _autocorrelate(self, sequence):
        if self.new_autocorr:
            N = len(sequence)
            divisors = np.full(2*N - 1, np.nan)
            for m in range(-N+1, 0):
                divisors[m + N - 1] = (N + m)
            for m in range(0, N):
                divisors[m + N - 1] = (N - m)
            corr = scp.signal.correlate(sequence, sequence, mode='full')
            corr /= divisors
            corr_orig = corr
            # Crop to 'same' size
            excess = len(corr) - len(sequence)
            # For now, at least, ensure we get an odd number of points.
            # Corr has odd length, so make sure we remove an even number of pts
            if excess % 2 == 1:
                excess += 1
            corr = corr[excess//2 : -excess//2]
            return corr
        return scp.signal.correlate(sequence, sequence, mode='full') / sequence.size
    
    def _calculate_fft_ps(self, data_list):
        """Calculates a spectrum directly as (FFT(data))**2"""
        
        powers = []
        frequencies = []
        
        w_sum = np.sum(self.weights)
        for v, w in zip(data_list, self.weights):
            # np.abs() handles complex values correctly.
            # Division by the array size correctly normalizes things.
            # n.b. the w/w_sum term here, along with the summing in
            # self._redistribute, makes this a weighted mean.
            powers.append(np.abs(np.fft.rfft(v) / v.size)**2 * w / w_sum)
            # We're using rfft, which only gives the positive frequencies.
            # The negative frequencies have the same power, so account
            # for that. Make sure not to double the zero-frequency term.
            powers[-1][1:] *= 2
            
            frequencies.append(np.fft.rfftfreq(v.size, self.dt))
        
        # OK, now we've got to average things together, but we're not guaranteed
        # equally-sized power spectra. (In fact, we won't have that at all unless
        # all the input velocity series are equally-sized.) In addition, we won't
        # even have the same frequency values for each spectrum, so we'll have to
        # redistribute power into a final set of bins.
        
        frequencies, powers = self._redistribute(frequencies, powers)
        
        return frequencies, powers
    
    def _redistribute(self, xs, ys):
        """Rebins data
        
        Given a list of x arrays (bin centers) and a list of y arrays (bin values),
        a new set of bins is created using the average of the unique bin spacings
        in the input histograms and covering the full domain of input bin centers.
        All input bin values are then redistributed proportionally according to the
        overlap proportion between an input bin and the output bin ranges."""
        min_x = np.inf
        max_x = -np.inf
        spacings = []
        for x in xs:
            spacing = x[1] - x[0]
            min_x = min(min_x, np.min(x) - spacing / 2)
            max_x = max(max_x, np.max(x) + spacing / 2)
            spacings.append(x[1] - x[0])
        
        spacings = np.unique(spacings)
        if len(spacings) == 1:
            return xs[0], ys[0]
        
        spacing = np.mean(spacings)
        min_center = min_x + spacing / 2
        max_center = max_x - spacing / 2
        target_n = (max_center - min_center) / spacing
        new_x = np.linspace(min_center, max_center, np.ceil(target_n).astype(int))
        #print("Using a spacing of {:.3g} for {} target bins".format(spacing, len(new_x)))
        
        new_y = np.zeros_like(new_x)
        
        for xx, yy in zip(xs, ys):
            # index of "target" bin receiving power
            idx = 0
            
            # target bin width
            tar_dx = new_x[1] - new_x[0]
            
            # source bin width
            src_dx = xx[1] - xx[0]
            
            #for x, y in zip(xx, yy):
            #for i in range(len(xx)):
            i = 0
            while True:
                # left/low and right/high edge of taget bin
                tar_l = new_x[idx] - tar_dx/2
                tar_r = new_x[idx] + tar_dx/2
                
                x = xx[i]
                y = yy[i]
                
                # left/low and right/high edge of source bin
                src_l = x - src_dx/2
                src_r = x + src_dx/2
                
                overlap = min(src_r, tar_r) - max(src_l, tar_l)
                
                new_y[idx] += overlap / src_dx * y
                
                if src_r == tar_r:
                    # The source bin's right edge coincides with the target
                    # bin's right edge, so we move on to the next source and
                    # target bins
                    idx += 1
                    i += 1
                elif src_r > tar_r:
                    # The source bin runs beyond the right edge of the target
                    # bin, so advance our target-bin index and repeat the loop
                    # with the same source bin
                    idx += 1
                else: # src_r < tar_r
                    # The source bin has been fully redistributed, so advance
                    # our source-bin index and repeat the loop with the same
                    # target bin
                    i += 1
                
                if i == len(xx):
                    # We've finished all the source bins
                    break
        
        return new_x, new_y
                
    def _make_FFT_continuous(self, freq, power):
        df = freq[1] - freq[0]
        
        return power / df
    
    def _calculate_WK(self, data_list, data_is_autocorr=False):
        """Calculates a spectrum via the Wiener-Khinchin Theorem"""
        # Start by autocorrelating each velocity sequence
        if data_is_autocorr:
            corrs = data_list
        else:
            corrs = []
            for v in data_list:
                # Select the non-NaN portion of the sequence
                # (NaNs show up when BPs leave the bounds of a ROUGH simulation)
                c = np.isfinite(v)
                autocorr = self._autocorrelate(v[c])
                corrs.append(autocorr)
        
        # AHH! Not every autocorrelation will be the same length! Let's pad
        # them out with NaNs.
        # Let's make a big array of NaNs, big enough to hold everything
        size = np.max([len(sequence) for sequence in corrs])
        all_x = np.full((len(corrs), size), np.nan, dtype=np.cdouble)
        all_w = np.full((len(corrs), size), np.nan)
        
        # Let's place each autocorrelation into a row of that array, putting
        # everything after and including the autocorrelation's
        # maximum at the beginning, and putting everything before the
        # maximum at the end of the row. The middle will be NaNs for
        # autocorrelations less than the maximum length. It's OK to offset
        # the position of things, since we'll do an FFT next, and that
        # implicitly assumes the input data is periodic.
        for i, (corr, w) in enumerate(zip(corrs, self.weights)):
            # Each sequence has an odd length, so this will find the middle
            max = corr.size // 2
            all_x[i, 0:max+1] = corr[max:]
            all_x[i, -max:] = corr[:max]
            
            all_w[i, 0:max+1] = w
            all_w[i, -max:] = w
        
        sum_w = np.sum(all_w, axis=0, where=np.isfinite(all_w))
        all_w /= sum_w
        all_x *= all_w
        
        # Now that the autocorrelations are aligned, let's average along every
        # column just those points that are not NaNs.
        # n.b. this sum is part of a weighted mean---see the w terms above
        corr = np.sum(all_x, axis=0, where=np.isfinite(all_x))
        n_samples = np.sum(np.isfinite(all_x), axis=0)
        
        # OK! Now we can do the FFT
        power = np.fft.fft(corr) / corr.size
        # For complex input sequences, we can have complex values at this
        # point. Taking the magnitude is a part of the "normal" way of making
        # a power spectrum, so...
        power = np.abs(power)
        
        # Now fold the negative frequencies onto the positive frequencies
        power = np.fft.fftshift(power)
        # This is the location of the zero frequency, for both even- and odd-
        # length arrays.
        cut = len(power)//2
        power[cut+1:] += power[:cut][::-1]
        power = power[cut:]
        
        return power, corr, n_samples
    
    def _generate_chitta_autocorr(self, use_delta=False):
        # fitted values for x dimension
        #a = 0.02
        #b = 1.77
        #Δ = 0.75
        #τ = 30.00
        #κ = 0.70
        
        # For power spectrum, use a=0 and b, τ, κ mean of x and y
        a = 0
        b = (1.77 + 1.45) / 2
        Δ = 0.75
        τ = (30.00 + 22.50) / 2
        κ = (0.70 + 0.90) / 2
        
        # Generate delay times to sample at
        dt = 5
        t = np.arange(-105, 105 + 1, dt) # +1 to include upper endpoint
        # Generate delay time index values
        n = t / dt
        
        c = a + b / (1 + (np.abs(t) / τ)**κ)
        
        if use_delta:
            mid = int((c.size - 1) / 2)
            c[mid] += Δ
            c[mid + 1] -= Δ/2
            c[mid - 1] -= Δ/2
        
        return t, c
    
    def velocity_histogram(self, show=True, title="Horiz velocity histogram"):
        components = [np.concatenate(x)**2 for x in self.time_series]
        v = np.sqrt(np.sum(components, axis=0))
        
        hist, edges = np.histogram(v, bins=100)
        
        centers = (edges[1:] + edges[0:-1])/2 / 1e5 #cm -> km
        
        plt.plot(centers, hist)
        plt.title(title)
        plt.ylabel("# Samples")
        plt.xlabel("|V$_{horiz}$| (km/s)")
        
        if show:
            plt.show()
        
    def generate_spectrum(self, print_power=True, smoothing=0, also_fft=False):
        also_fft = also_fft or print_power
        if also_fft:
            fft_powers = []
            for ts in self.time_series:
                frequencies, fft_p = self._calculate_fft_ps(ts)
                fft_powers.append(fft_p)
        
        powers = []
        corrs = []
        n_samples = []
        for ts in self.time_series:
            power, corr, n = self._calculate_WK(ts)
            powers.append(power)
            corrs.append(corr)
            n_samples.append(n)
        
        WK_frequencies = np.fft.rfftfreq(corrs[0].size, self.dt)
        
        if print_power:
            print("Power: {:e} WK, {:e} fft".format(
                np.sum([power for power in powers]),
                np.sum([power for power in fft_powers])))
        
        if smoothing:
            powers = [self._smooth(power.real, smoothing) for power in powers]
        
        powers = [self._make_FFT_continuous(WK_frequencies, power)
                  for power in powers]
        
        # Remove the zero-frequency term, for log-log plotting
        WK_frequencies = WK_frequencies[1:]
        zero_freqs = [power[0] for power in powers]
        powers = [power[1:] for power in powers]
        
        res = dict()
        res['powers'] = powers
        res['zero_freqs'] = zero_freqs
        res['corrs'] = corrs
        res['n_samples'] = n_samples
        res['WK_frequencies'] = WK_frequencies
        
        if also_fft:
            frequencies = frequencies[1:]
            fft_powers = [power[1:] for power in fft_powers]
            fft_powers = [self._make_FFT_continuous(frequencies, power)
                          for power in fft_powers]
            res['frequencies'] = frequencies
            res['fft_powers'] = fft_powers
        
        return res
    
    def overlay_chitta(self, do_slope=True, print_power=True, label='Chitta'):
        t, autocorr = self._generate_chitta_autocorr()
        power, corr, n_samples = self._calculate_WK([autocorr], data_is_autocorr=True)
        freq = np.fft.rfftfreq(autocorr.size, 5)
        freq = freq[1:]
        power = power[1:]
        
        # To cgs
        unit_conv = 1e10
        power *= unit_conv
        
        # Account for x and y
        power *= 2
        
        if print_power:
            print("Chitta Power: {:e}".format(np.sum(np.abs(power).real)))
        
        power = self._make_FFT_continuous(freq, power)
        
        x, xlabel = self._get_x(freq)
        y = 1e-10*power.real
        
        plt.loglog(x, y, label=label, color='black', linestyle='dotted', linewidth=2)
        plt.ylabel("Power spectrum (km $^2$ s $^{-2}$ Hz $^{-1}$)")
        plt.xlabel("Frequency (Hz)")
        
        if do_slope:
            self._calc_slope(x, y)
    
    def overlay_cranmer(self, print_power=True):
        # Cranmer stuff
        σ2w = 0.8 * (1e5)**2 # km^2/s^2 --> cm^2/s^2
        τw = 60 # s
        τj = 20 # s
        Δt = 360 # s
        σ2j = 3**2 * (1e5)**2 # km^2/s^2 --> cm^2/s^2
        
        def cranmer_autocorr(t):
            Cw = σ2w / (1 + (t/τw)**2)
            Cj = np.sqrt(np.pi/2) * σ2j * τj / Δt * np.exp(
                    -t**2 / 2 / τj**2)
            return Cw, Cj
        def cranmer_power(f):
            ω = 2*np.pi * f
            # Skip his rho? He has a rho at the beginning of the P expression
            Pw = σ2w * τw * np.exp(-τw * ω)
            Pj = σ2j * τj**2 / Δt * np.exp(-ω**2 * τj**2 / 2)
            return Pw + Pj
        t = np.arange(-105, 105 + 1, 2)
        Cw, Cj = cranmer_autocorr(t)
        powerW = self._calculate_WK([Cw], data_is_autocorr=True)[0]
        powerJ = self._calculate_WK([Cj], data_is_autocorr=True)[0]
        freq = np.fft.rfftfreq(Cw.size, t[1] - t[0])
        freq = freq[1:]
        power = powerW + powerJ
        #power = powerW
        power = power[1:]
        power *= 2
        power = np.abs(power).real
        
        #power = cranmer_power(freq)
        ## Account for x and y
        #power *= 2
        
        #df = freq[1] - freq[0]
        #print("Cranmer Power: {:e}".format(df * np.sum(np.abs(power).real)))
        if print_power:
            print("Cranmer Power: {:e}".format(np.sum(np.abs(power).real)))
        power = self._make_FFT_continuous(freq, power)
        
        x, xlabel = self._get_x(freq)
        #plt.loglog(x, power, '--', label="Cranmer", color='black')
        
    def _calc_slope(self, x, y):
        for f in [0.001, 0.01]:
            plt.axvline(f, color="#CCCCCC", zorder=-30)
            
            # Find nearest frequency bin index
            idx = np.argmin(np.abs(x - f))
            if idx in (0, x.size-1):
                continue
            x1 = np.max((idx-3, 0))
            x2 = np.min((idx+3, x.size))
            y1 = np.max((idx-3, 0))
            y2 = np.min((idx+3, y.size))
            
            xx = np.log10(x[x1:x2])
            yy = np.log10(y[y1:y2])
            
            m, *_ = scp.stats.linregress(xx, yy)
            
            #m = np.log10(y2/y1) / np.log10(x2/x1)
            print("Slope at {}: {}".format(f, m))
            
            # Plot little sample slope lines
            #xx1 = xx[0]
            #xx2 = xx[-1]
            #yy1 = yy[0]
            #yy2 = yy1 + m * (xx2-xx1)
            #plt.loglog( [10**xx1, 10**xx2], [10**yy1, 10**yy2], 'b')
        
        
    def plot(self, show=False, title="Power Spectrum", clear=True,
            print_power=False, smoothing=0, overlay_chitta=False,
            label=None, color=None, show_corr=False, linestyle=None,
            scale_factor=1e-10, mark_slope_measurements=False, also_fft=False,
            chitta_label='Chitta', linearize=False, **kwargs):
        """Calculates and plots the power spectrum.
        
        Each velocity sequence accumulated (via the add() function) is
        auto-correlated. The autocorrelations are aligned based on their
        maximum values and averaged together (using, for each offset time, only
        those autocorrelations with valued values for the offset time). This
        averaged autocorrelation is passed through an FFT to get a power
        spectrum (via the Wiener-Khinchin theorem). This is all done separately
        for x and y, and both power spectra are returned.
        
        Parameters:
            show: whether to call plt.show()
            title: the title of the plot
            clear: whether to clear the accumulated list of velocity sequences
                   after using the list to compute a power spectrum
            print_power: whether to print the real and imaginary power"""
        
        res = self.generate_spectrum(print_power, smoothing, also_fft=also_fft)
        
        if show_corr:
            plt.subplot(121)
            x = np.arange(len(res['corrs'][0]))
            x -= x[x.size//2]
            for corr, label in zip(res['corrs'], ('X', 'Y', 'Z')):
                plt.plot(x, corr, label=label)
            plt.title("Autocorrelation")
            plt.legend()
            
            # Just hack things up here to ensure consistent plotting, bounds,
            # visibility, etc. between different plots
            #plt.ylim(-.1e9, 1e9)
            xmin, xmax = plt.xlim()
            diff = xmax - xmin
            plt.xlim(xmin - 0.05 * diff, xmax + 0.05 * diff)
            
            plt.subplot(122)
        
        x, xlabel = self._get_x(res['WK_frequencies'])
        y = scale_factor * np.sum([p.real for p in res['powers']], axis=0)
        
        if linearize:
            x = np.log10(x)
            y = np.log10(y)
            plot_fcn = plt.plot
        else:
            plot_fcn = plt.loglog
        
        line = plot_fcn(x, y, label=label, color=color,
                 linestyle=linestyle, linewidth=2, **kwargs)
        res['plotted_x'] = x
        res['plotted_y'] = y
        res['plotted_line'] = line
        
        if also_fft:
            x, xlabel = self._get_x(res['frequencies'])
            y = scale_factor * np.sum(
                    [p.real for p in res['fft_powers']], axis=0)
            
            if linearize:
                x = np.log10(x)
                y = np.log10(y)
                plot_fcn = plt.plot
            else:
                plot_fcn = plt.loglog
            
            plot_fcn(x, y, label=label + " (fft)", color=color,
                     linestyle=linestyle, linewidth=2, **kwargs)
        
        
        if mark_slope_measurements:
            self._calc_slope(x, y)
        
        if overlay_chitta:
            self.overlay_chitta(do_slope=print_power, print_power=print_power,
                    label=chitta_label)
        
        if label is not None:
            plt.legend(loc="lower left", framealpha=1)
        plt.xlabel(xlabel)
        plt.ylabel("Power spectrum (km $^2$ s $^{-2}$ Hz $^{-1}$)")
        plt.title(title)
        
        if show:
            plt.show()
        
        if clear:
            self.time_series = []
            self.weights = []
        return res
    
    def _get_x(self, x):
        if self.plot_period:
            return 1/x, "Period (s)"
        else:
            return x, "Frequency (Hz)"
    
    def _smooth(self, y, window_radius=3):
        """Applies boxcar smoothing to the signal in log space
        """
        if window_radius is True:
            # Use default for backwards compat
            window_radius = 3
        window_size = 1 + 2 * window_radius
        
        logy = np.log10(y)
        
        window = scp.signal.boxcar(window_size)
        result = np.zeros_like(y)
        result[window_radius:-window_radius] = np.convolve(logy, window, mode='valid') / window.size
        
        # For the edges, just do an asymmetric boxcar
        for i in range(0, window_radius):
            result[i] = np.mean(logy[0:window_radius + i + 1])
        for i in range(len(result)-window_radius-1, len(result)):
            result[i] = np.mean(logy[i-window_radius:])
        
        result = 10**result
        return result

if __name__ == "__main__":
    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        files = os.listdir('.')

    for filename in files:
        if len(filename) < 4 or filename[-4:] != '.pkl':
            continue

        data = pickle.load(open(filename, 'rb'))

        power_spectrum(data, show=True)
