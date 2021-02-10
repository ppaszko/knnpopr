import getopt
import sys
from smtpd import usage
import numpy as np
loaded_data = np.loadtxt('prediction.csv', delimiter=' ', dtype='float')


new=np.delete(loaded_data, -1, 1)

np.savetxt('prediction2.csv',new, delimiter=' ')