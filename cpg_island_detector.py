import numpy as np
from hmmlearn import hmm
import input_handler as inhd

states = ["A+", "C+", "G+", "T+", "A-", "C-", "G-", "T-"]
n_states = len(states)

observations = ["a", "c", "g", "t"]
n_observations = len(observations)

start_probability = np.array([0.0725193,
0.1637630,
0.1788242,
0.0754545,
0.1322050,
0.1267006,
0.1226380,
0.1278950])

transition_probability = np.array([
  [0.1762237+(1-0.9989997),
0.2682517,
0.4170629,
0.1174825,
0.0035964,
0.0054745,
0.0085104,
0.0023976
],
  [0.1672435+(1-0.9989996),
0.3599201,
0.2679840,
0.1838722,
0.0034131,
0.0073453,
0.0054690,
0.0037524
],
   [0.1576223+(1-0.9989997),
0.3318881,
0.3671328,
0.1223776,
0.0032167,
0.0067732,
0.0074915,
0.0024975
],
    [0.0773426+(1-0.9989957),
0.3475514,
0.375940,
0.1781818,
0.0015784,
0.0070929,
0.0076723,
0.0036363
],
     [0.0002997+(1-0.9986041),
0.0002047,
0.0002837,
0.0002097,
0.299005,
0.2045904,
0.2844305,
0.2095804
],
      [0.0003216+(1-0.9989996),
0.0002977,
0.0000769,
0.0003016,
0.3213566,
0.2974045,
0.0778441,
0.3013966
],
       [0.0001768+(1-0.9989996),
0.0002387,
0.0002917,
0.0002917,
0.1766463,
0.2385224,
0.2914165,
0.2914155
],
        [0.0002477+(1-0.9989995),
0.0002457,
0.0002977,
0.0002077,
0.2475044,
0.2455084,
0.2974035,
0.2075844
]
])

emission_probability = np.array([
  [1,0,0,0],
  [0,1,0,0],
  [0,0,1,0],
  [0,0,0,1],
  [1,0,0,0],
  [0,1,0,0],
  [0,0,1,0],
  [0,0,0,1]
])

model = hmm.MultinomialHMM(n_components=n_states)
model.startprob_ = start_probability
model.transmat_  = transition_probability
model.emissionprob_ = emission_probability

in_text = inhd.get_gene_sequence("gene.txt")
logprob, out_text = model.decode(in_text, algorithm="viterbi")
print("Input:    %s"%" ".join([observations[x[0]] for x in in_text]))
print("Output: %s"%" ".join([states[x][-1] for x in out_text]))
