import math
import numpy as np
from scipy.interpolate import splrep, BSpline

class TransferFunction:

    """
    Create a transfer function coding object. The input (camera RAW) signal will be between 0 and 2**bits-1.
    The encoded values will be between 0 and 1.
    """
    def __init__(self, bits):
        self.bits = bits

    def decode(self,V):
        L_float = self.decode_float(V)
        return (L_float+0.5).astype(int)  # add 0.5 for rounding

    def decode_float(self, V):
        raise NotImplementedError()


class TF_linear(TransferFunction):

    """
    Encode linear camera values (irradiance) given as integer values between 0 and 2**bits-1.
    The encoded values V will be in the 0-1 range.
    """
    def encode(self, L):
        V = L.astype(float)/(2**self.bits-1)
        return V

    """
    Decode values produced by "encode" method into linear camera values (irradiance). The camera
    linear values are in the range between 0 and 2**bits-1.
    The encoded values V must be in the range from 0 to 1.
    """
    def decode_float(self, V):
        L = V * (2**self.bits-1)
        return L


class TF_log(TransferFunction):
    epsilon=0.01
    def encode(self, L):
        V = np.log2(L.astype(float)+self.epsilon) / math.log2(2**self.bits-1)
        return V

    def decode_float(self, V):
        L = np.power(2, V*math.log2(2**self.bits-1))-self.epsilon
        return L


class TF_gamma(TransferFunction):
    gamma=2.2

    def encode(self, L):
        V = np.power(L.astype(float)/(2**self.bits-1), 1/self.gamma)
        return V

    def decode_float(self, V):
        L = np.power(V,self.gamma)*(2**self.bits-1)
        return L


class TF_PQ(TransferFunction):
    #Lmax = 10000
    n    = 0.15930175781250000
    m    = 78.843750000000000
    c1   = 0.83593750000000000
    c2   = 18.851562500000000
    c3   = 18.687500000000000

    def __init__(self, bits):
        self.bits = bits
        self.max_val = 2**self.bits-1
        self.min_val = self._abs_encode( 1 )

    def _abs_encode(self, L):
        im_t = np.power(L/self.max_val,self.n)
        V  = np.power((self.c2*im_t + self.c1) / (1+self.c3*im_t), self.m)
        return V

    def encode(self, L):
        return (self._abs_encode(L) - self.min_val)/(1 - self.min_val)

    def decode_float(self, V):
        V_rescaled = V *  (1 - self.min_val) + self.min_val
        im_t = np.power(np.maximum(V_rescaled,0),1/self.m)
        L = self.max_val * np.power(np.maximum(im_t-self.c1,0)/(self.c2-self.c3*im_t), 1/self.n)
        return L

class TF_VarianceStabilizing(TransferFunction):

    def __init__(self, bits, k, std_read, std_adc, gain=1):
        self.bits = bits
        self.k = k
        self.var_read = std_read**2
        self.var_adc = std_adc**2
        self.gain = gain
        self.min_v = self._abs_encode(0)
        self.max_v = self._abs_encode(2**bits-1)

    def _abs_encode(self,L):
        return 2*np.sqrt(L*self.gain*self.k + self.var_read*self.gain**2 + self.var_adc)/(self.gain*self.k)

    def encode(self, L):
        V = (self._abs_encode(L)-self.min_v)/(self.max_v-self.min_v)
        return V

    def decode_float(self, V):
        V2 = np.power(V*(self.max_v-self.min_v)+self.min_v,2)
        L = (0.25*V2*self.gain**2 * self.k**2 - self.var_read*self.gain**2 - self.var_adc)/(self.gain*self.k)
        return L


"""
Logarithmic transfer function that cuts anything below a certain signal-to-noise ratio
"""
class TF_LogSNRThr(TransferFunction):

    """
    snr_db - signal-to-ratio in dB units. For example snr_db=0 or snr_db=5
    std_read - camera readout noise
    std_adc - camera ADC noise
    """
    def __init__(self, bits, snr_db, k, std_read, std_adc, gain=1):
        self.bits = bits
        self.k = k
        snr_lin = 10**(snr_db/10)
        self.Y_thr = 0.5 * snr_lin * (math.sqrt(snr_lin**2 * gain**2 * k**2 + 4*gain**2*std_read**2 + 4*std_adc**2) + snr_lin*gain*k)
        self.min_v = math.log(self.Y_thr)
        self.max_v = math.log(2**bits-1)

    def encode(self,L):
        below_noise_floor = (L <= self.Y_thr)
        V = (np.log(L)-self.min_v)/(self.max_v - self.min_v)
        V[below_noise_floor] = 0
        return V

    def decode_float(self, V):
        v_is_0 = (V == 0)
        L = np.exp(V*(self.max_v-self.min_v)+self.min_v)
        L[v_is_0] = self.Y_thr
        return L


"""
Two-segment logarithmic transfer function that assignes lower contrast to the content below certain noise level
"""
class TF_LogSNRThr2Seg(TransferFunction):

    """
    snr_db - signal-to-ratio in dB units. For example snr_db=0 or snr_db=5
    std_read - camera readout noise
    std_adc - camera ADC noise
    """
    def __init__(self, bits, snr_db, k, std_read, std_adc, gain=1):
        self.bits = bits
        self.k = k
        snr_lin = 10**(snr_db/10)
        self.Y_thr = 0.5 * snr_lin * (math.sqrt(snr_lin**2 * gain**2 * k**2 + 4*gain**2*std_read**2 + 4*std_adc**2) + snr_lin*gain*k)
        l = math.log2(self.Y_thr)  # stop below noise level
        h = math.log2((2**bits-1)/(self.Y_thr))  # stops above noise level
        self.l = l
        self.s_h = 1/(0.5*l + h)
        self.s_l = 0.5*self.s_h
        self.V_thr = l*self.s_l


    def encode(self,L):
        bnf = (L <= self.Y_thr)  # Below noise floor
        anf = np.logical_not(bnf) # Above noise floor
        bnf = np.logical_xor(bnf,L==0)
        V = np.zeros(L.shape)
        V[bnf] = np.log2(L[bnf])*self.s_l
        V[anf] = (np.log2(L[anf])-np.log2(self.Y_thr))*self.s_h + self.V_thr
        return V

    def decode_float(self, V):
        bnf = (V <= self.V_thr)  # Below noise floor
        anf = np.logical_not(bnf) # Above noise floor
        bnf = np.logical_xor(bnf,V==0)

        L = np.zeros(V.shape)
        L[bnf] = np.power(2,V[bnf]/self.s_l)
        L[anf] = np.power(2,(V[anf]-self.V_thr)/self.s_h+np.log2(self.Y_thr))
        return L


"""
Two-segment logarithmic transfer function that transitions smoothly
"""
class TF_LogSNRThr2SegSmooth(TransferFunction):

    """
    snr_db_start - signal-to-noise ratio in dB units where the transition from low slope begins
    snr_db_end - signal-to-noise ratio where transition to higher slope is completed
    """
    def __init__(self, bits, snr_db_start, snr_db_end, k, std_read, std_adc, gain=1):
        self.bits = bits
        self.k = k
        snr_lin = 10**(np.array([snr_db_start, (snr_db_start+snr_db_end)/2, snr_db_end])/10)
        self.Y_thr = 0.5 * snr_lin * (np.sqrt(snr_lin**2 * gain**2 * k**2 + 4*gain**2*std_read**2 + 4*std_adc**2) + snr_lin*gain*k)
        l = math.log2(self.Y_thr[1])
        h = math.log2((2**bits-1)/(self.Y_thr[1]))
        self.s_h = 1/(0.5*l + h)
        self.s_l = 0.5*self.s_h
        self.V_thr = np.log2(self.Y_thr)*self.s_l
        self.V_thr[2] = (np.log2(self.Y_thr[2]) - math.log2(self.Y_thr[1]))*self.s_h + self.V_thr[1]

        # Find cubic polynomial that passes through the 2 thresholds
        # and has slopes s_l and s_h at the respective thresholds
        x1, x2 = np.log2(self.Y_thr[::2])
        y1, y2 = self.V_thr[::2]
        A = np.array([[3*x1**2, 2*x1, 1, 0],
        			  [3*x2**2, 2*x2, 1, 0],
        			  [x1**3, x1**2, x1, 1],
        			  [x2**3, x2**2, x2, 1]])
        B = np.array([self.s_l, self.s_h, y1, y2])
        self.poly = np.poly1d(np.linalg.solve(A, B))

    def encode(self, L):
        bnf = (L <= self.Y_thr[0])                      # Below noise transition
        anf = (L > self.Y_thr[2])                       # Above noise transition
        inf = np.logical_not(np.logical_or(bnf, anf))   # In the noise transition
        bnf = np.logical_xor(bnf, L==0)
        V = np.zeros(L.shape)
        V[bnf] = np.log2(L[bnf])*self.s_l
        V[anf] = (np.log2(L[anf]) - math.log2(self.Y_thr[1]))*self.s_h + self.V_thr[1]
        V[inf] = self.poly(np.log2(L[inf]))
        return V

    def decode_float(self, V):
        bnf = (V <= self.V_thr[0])                      # Below noise transition
        anf = (V >= self.V_thr[2])                      # Above noise transition
        inf = np.logical_not(np.logical_or(bnf, anf))   # In the noise transition
        bnf = np.logical_xor(bnf, V==0)

        L = np.zeros(V.shape)
        L[bnf] = np.power(2, V[bnf]/self.s_l)
        L[anf] = np.power(2, (V[anf]-self.V_thr[1])/self.s_h+np.log2(self.Y_thr[1]))
        L[inf] = np.power(2, [(self.poly - v).roots[1] for v in V[inf]])

        return L
